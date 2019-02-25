import re
import numpy as np
import glob
import os
import argparse
import subprocess
from itertools import izip, product
from struct import unpack
try:
    import tqdm
except ImportError:
    tqdm = None

def line_count(file_name):
    ''' Returns the number of lines in file_name by calling wc. '''
    wc_output = subprocess.check_output(['wc', '-l', file_name])
    count, _ = wc_output.strip().split()
    count = int(count)
    return count

def custom_progress_bar(max_val):
    if progressbar:
        max_str = str(len(str(max_val)))
        format_string = '%(value)' + max_str + 'd / %(max)d'
        widgets = [progressbar.Bar('='),
                   ' ',
                   progressbar.FormatLabel(format_string),
                   ' ',
                   progressbar.ETA(),
                  ]
        progress_bar = progressbar.ProgressBar(widgets=widgets, maxval=max_val).start()
    else:
        # If progressbar isn't installed, create a dummy class that will look like a progressbar but do nothing
        class Dummy(object):
            def update(self, number):
                pass

            def finish(self):
                pass

        progress_bar = Dummy()

    return progress_bar

def load_locations(lane_dir):
    ''' Loads cluster location information from .locs files of the form
        s_1_{tile number}.locs in lane_dir. 
        
        Returns a dict with key-value pairs of the form 
        {tile number: location dict}, where location dict has key-value pairs
        of the form {x, y: cluster number}.
        
        .locs format description, copied from picard documentation
        bytes 0-3 : (int?) Version number (1) 
        bytes 4-7 : 4 byte float equaling 1.0
        bytes 8-11 : unsigned int numClusters
        bytes 12-15: : X coordinate of first cluster (32-bit float)
        bytes 16-19: : Y coordinate of first cluster (32-bit float)
        The remaining bytes of the file store the X and Y coordinates of the remaining clusters.
    '''
    def transform_coordinate(f):
        ''' Transform coordinate from .locs system to fastq system, as per
            http://www.biostars.org/p/51681/
        '''
        return int(round(10 * f)) + 1000
    
    locs_files = sorted(glob.glob('{0}/*.locs'.format(lane_dir)))
    
    locs = {}
    for i, locs_file in tqdm.tqdm_notebook(enumerate(locs_files), desc='Tiles', total=len(locs_files)):
        root, ext = os.path.splitext(locs_file)
        tile = int(root.split('_')[-1])
        
        locs_handle = open(locs_file, 'rb')
        
        version, = unpack('i', locs_handle.read(4))
        if version != 1:
            message = 'LOCS version number of {0}, expected 1'.format(version)
            raise ValueError(message)
        
        one, = unpack('f', locs_handle.read(4))
        if one != 1.0:
            message = 'LOCS float check of {0:.2f}, expected 1.0'.format(one)
            raise ValueError(message)
        
        num_clusters, = unpack('i', locs_handle.read(4))
        
        locs[tile] = {}

        #for c in tqdm.tqdm_notebook(xrange(num_clusters), desc=str(tile), total=num_clusters, leave=False):
        for c in xrange(num_clusters):
            eight_bytes = locs_handle.read(8)
            
            if len(eight_bytes) != 8:
                message = 'LOCS file contains less clusters than indicated in header'
                raise ValueError(message)
            
            x, y = unpack('ff', eight_bytes)
            x, y = transform_coordinate(x), transform_coordinate(y)
            locs[tile][x, y] = c

        if locs_handle.read(1):
            message = 'LOCS file contains more data than indicated in header'
            raise ValueError(message)
        
    return locs

def load_intensities(lane_dir, tile, num_clusters):
    ''' Loads intensity values from .cif files located in subdirectories of
        lane_dir. Returns a 3D array indexed by 
        (cluster number, base identity, cycle number).
    
        .cif format, copied from HCS 1.4/RTA 1.12 Theory of Operation 
        bytes 0-2: CIF
        byte 3: Version number (1)
        byte 4: Precision. Can be 1 for a file storing intensities as signed
                bytes, 2 for values stored as signed 2-byte integers, or 4 for
                values stored as 4-byte floating-point values. Normal .cif
                files use 2 bytes of precision.
        bytes 5-6: Cycle (unsigned short)
        bytes 7-8: 1 (unsigned short)
        bytes 9-12: Cluster count (unsigned int)
        The remainder of the file stores the A intensities, then C, then G,
        then T. The intensities for each channel take up 
        (Precision * ClusterCount) bytes.
    '''
    cycle_dirs = glob.glob('{0}/C*'.format(lane_dir))
    cycle_pattern = re.compile(r'C(\d+)\.1')
    extract_cycle = lambda s: int(cycle_pattern.search(s).group(1))
    cycles = sorted([extract_cycle(cycle_dir) for cycle_dir in cycle_dirs])
    
    intensities = np.zeros(shape=(num_clusters, 4, len(cycles)), dtype=np.int)
    
    progress_bar = custom_progress_bar(len(cycles))
    for cycle in cycles:
        tile_file = '{0}/C{1}.1/s_1_{2}.cif'.format(lane_dir,
                                                    cycle,
                                                    tile,
                                                   )
        with open(tile_file, 'rb') as tile_handle:
            cif = tile_handle.read(3)
            if cif != 'CIF':
                message = '\'CIF\' expected in CIF header, got {0}'.format(cif)
                raise ValueError(message)
            
            version, = unpack('b', tile_handle.read(1))
            if version != 1:
                message = 'CIF version number of {0}, expected 1'.format(version)
                raise ValueError(message)
            
            precision, = unpack('b', tile_handle.read(1))
            if precision != 2:
                message = 'Support for CIF precision of {0} not implemented'.format(precision)
                raise NotImplementedError(message)

            cycle_claimed, = unpack('H', tile_handle.read(2))
            if cycle_claimed != cycle:
                message = 'Cycle of {0} claimed in CIF header, expected {1}'.format(cycle_claimed,
                                                                                    cycle,
                                                                                   )
                raise ValueError(message)

            # Next 2 bytes don't seem to agree with the standard. Ignore them.
            one = unpack('H', tile_handle.read(2))

            num_clusters_claimed, = unpack('I', tile_handle.read(4))
            if num_clusters_claimed != num_clusters:
                message = '{0} clusters claimed in CIF header, expected {1}'.format(num_clusters_claimed,
                                                                                    num_clusters,
                                                                                   )
                raise ValueError(message)

            # Order in .cif is ACGT
            format_string = '{0}h'.format(num_clusters)
            for base in range(4):
                all_clusters = tile_handle.read(num_clusters * 2)
                if len(all_clusters) != num_clusters * 2:
                    message = 'CIF file ended prematurely'
                    raise ValueError(message)
                      
                all_intensities = unpack(format_string, all_clusters)
                for cluster, intensity in enumerate(all_intensities):
                    intensities[cluster, base, cycle - 1] = intensity
            
            # Make sure file ended when it said it would
            if tile_handle.read(1):
                message = 'CIF file contains more data than indicated in header'
                raise ValueError(message)

        progress_bar.update(cycle)
    progress_bar.finish()

    return intensities

def intensities_array_to_strings(intensities_array):
    def base_to_string(base):
        strings = map(str, base)
        string = '\t'.join(strings) + '\n'
        return string
    
    def cluster_to_string(cluster):
        base_strings = map(base_to_string, cluster)
        string = ''.join(base_strings)
        return string
    
    num_clusters, _, _ = intensities_array.shape
    intensity_strings = []
    progress_bar = custom_progress_bar(num_clusters)
    for c, cluster in enumerate(intensities_array):
        string = cluster_to_string(cluster)
        intensity_strings.append(string)
        
        progress_bar.update(c + 1)
    progress_bar.finish()

    return intensity_strings

def make_intensity_files(fastq_files, lane_dir):
    name_delimiter = re.compile(r'[ @]')
    def parse_name_line(name_line):
        name = name_line.strip().lstrip('@')
        cluster_info, read_info = re.split(name_delimiter, name)
        tile, x, y = cluster_info.split(':')[-3:]
        tile, x, y = int(tile), int(x), int(y)
        return cluster_info, tile, x, y

    def split_int_file_name(fastq_file, tile):
        root, ext = os.path.splitext(fastq_file)
        int_file = '{0}_int_{1}.txt'.format(root, tile)
        return int_file

    def merged_int_file_name(fastq_file):
        root, ext = os.path.splitext(fastq_file)
        int_file = '{0}_int.txt'.format(root, tile)
        return int_file

    fastq_line_counts = {}
    print('Counting reads in fastq files...')
    progress_bar = custom_progress_bar(len(fastq_files))
    for i, fastq_file in enumerate(fastq_files):
        fastq_line_counts[fastq_file] = line_count(fastq_file) / 4
        progress_bar.update(i + 1)
    progress_bar.finish()
    
    print('Loading cluster locations in tiles...')
    locations = load_locations(lane_dir)
    cluster_counts = {tile:len(locations) for tile, locations in locations.items()}
    
    for t, tile in enumerate(locations):
        print('Processing tile {0} ({1}/{2})'.format(tile, t + 1, len(locations)))
        print('Loading intensities from .cifs...')
        intensities = load_intensities(lane_dir, tile, cluster_counts[tile])
        print('Converting to strings...')
        intensity_strings = intensities_array_to_strings(intensities)
        
        for fastq_file in fastq_files:
            print('Assigning intensities to {0}'.format(fastq_file))
            
            intensities_file_name = split_int_file_name(fastq_file, tile)

            fastq_handle = open(fastq_file)
            fastq_reads = izip(*[fastq_handle]*4)
            name_lines = (name_line for name_line, _, _, _ in fastq_reads)
            
            with open(intensities_file_name, 'w') as intensities_handle:
                progress_bar = custom_progress_bar(fastq_line_counts[fastq_file])
                for l, name_line in enumerate(name_lines):
                    cluster_info, read_tile, x, y = parse_name_line(name_line)
                    
                    if read_tile == tile:
                        if (x, y) not in locations[tile]:
                            # Attempt to deal with bizarre rounding irregularities
                            valid_nearby = []
                            offsets = [-1, 0, 1]
                            for x_offset, y_offset in product(offsets, offsets):
                                if (x + x_offset, y + y_offset) in locations[tile]:
                                    valid_nearby.append((x + x_offset, y + y_offset))
                            
                            if len(valid_nearby) != 1:
                                raise ValueError('Unable to resolve rounding irregularity')
                            
                            x, y = valid_nearby[0]
                        
                        cluster_index = locations[tile][x, y]
                        intensity_string = intensity_strings[cluster_index]
                        
                        intensities_handle.write(name_line)
                        intensities_handle.write(intensity_string)
                    
                    progress_bar.update(l + 1)
                progress_bar.finish()
                 
    for fastq_file in fastq_files:
        print('Merging tiles for {0}'.format(fastq_file))
        tile_handles = {tile: open(split_int_file_name(fastq_file, tile))
                        for tile in locations}
        
        merged_file = merged_int_file_name(fastq_file)
        
        fastq_handle = open(fastq_file)
        fastq_reads = izip(*[fastq_handle]*4)
        name_lines = (name_line for name_line, _, _, _ in fastq_reads)
        
        with open(merged_file, 'w') as merged_handle:
            progress_bar = custom_progress_bar(fastq_line_counts[fastq_file])
            
            for l, name_line in enumerate(name_lines):
                cluster_info, read_tile, x, y = parse_name_line(name_line)
                tile_handle = tile_handles[read_tile]
                
                tile_name_line = tile_handle.readline()
                if tile_name_line != name_line:
                    raise RuntimeError('Split files out of sync')

                merged_handle.write('@{0}\n'.format(cluster_info))
                for i in range(4):
                    merged_handle.write(tile_handle.readline())
                
                progress_bar.update(l + 1)
            progress_bar.finish()
        
        print('Removing tile-specific files...')
        for tile in locations:
            os.remove(split_int_file_name(fastq_file, tile))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('fastq_files',
                        help='comma-separated list of fastq file names to extract the intensities of the reads in',
                        type = lambda s: s.split(','),
                       )
    parser.add_argument('lane_dir',
                        help='directory containing .loc files for each tile and subdirectories for each cycle containing .cif files',
                       )
    args = parser.parse_args()

    make_intensity_files(args.fastq_files, args.lane_dir)
