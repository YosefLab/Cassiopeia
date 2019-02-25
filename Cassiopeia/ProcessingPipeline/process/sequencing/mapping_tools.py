import os
import tempfile
import logging
import subprocess
import threading
import shutil

import pysam
import numpy as np

from . import genomes, fasta, fastq

def build_bowtie2_index(index_prefix, sequence_file_names):
    bowtie2_build_command = ['bowtie2-build',
                             ','.join(sequence_file_names),
                             index_prefix,
                            ]
    subprocess.check_call(bowtie2_build_command)

def map_bwa(file_name, genome, sai_file_name, sam_file_name, error_file_name, threads=1):
    ''' Map reads in file_name to genome with bwa. '''
    index_location = '{0}/{1}'.format(bwa_indexes_location, genome)
    with open(sai_file_name, 'w') as sai_file, open(error_file_name, 'w') as error_file:
        bwa_aln_command = ['bwa', 'aln',
                           '-t', str(threads),
                           index_location,
                           file_name,
                          ]
        subprocess.check_call(bwa_aln_command, 
                              stdout=sai_file,
                              stderr=error_file,
                             )
    
    with open(sam_file_name, 'w') as sam_file, open(error_file_name, 'a') as error_file:
        bwa_samse_command = ['bwa', 'samse',
                             '-n', '100',
                             index_location,
                             sai_file_name,
                             file_name,
                            ]
        subprocess.check_call(bwa_samse_command,
                              stdout=sam_file,
                              stderr=error_file,
                             )
    
def map_paired_bwa(R1_file_name, R2_file_name,
                   genome,
                   R1_sai_file_name, R2_sai_file_name,
                   sam_file_name,
                   error_file_name,
                   threads=1,
                  ):
    ''' Map paired end reads in R1_file_name and R2_file_name to genome with bwa. '''
    index_location = '{0}/{1}'.format(bwa_indexes_location, genome)
    with open(R1_sai_file_name, 'w') as R1_sai_file, open(error_file_name, 'w') as error_file:
        bwa_aln_R1_command = ['bwa', 'aln', '-t', str(threads), index_location, R1_file_name]
        subprocess.check_call(bwa_aln_R1_command,
                              stdout=R1_sai_file,
                              stderr=error_file,
                             )
    with open(R2_sai_file_name, 'w') as R2_sai_file, open(error_file_name, 'w') as error_file:
        bwa_aln_R2_command = ['bwa', 'aln', '-t', str(threads), index_location, R2_file_name]
        subprocess.check_call(bwa_aln_R2_command,
                              stdout=R2_sai_file,
                              stderr=error_file,
                             )

    bwa_sampe_command = ['bwa', 'sampe',
                         '-n', '100',
                         '-r', '@RG\tID:0\tSM:6',
                         '-a', '1000',
                         index_location,
                         R1_sai_file_name, R2_sai_file_name,
                         R1_file_name, R2_file_name,
                        ]
    with open(sam_file_name, 'w') as sam_file, open(error_file_name, 'a') as error_file:
        subprocess.check_call(bwa_sampe_command,
                              stdout=sam_file,
                              stderr=error_file,
                             )

class DoNothing(object):
    def __enter__(self):
        pass

    def __exit__(self, exception_type, exception_value, exception_traceback):
        pass

class TemporaryFifo(object):
    def __init__(self, name='fifo'):
        self.name = name

    def __enter__(self):
        self.temp_dir = tempfile.mkdtemp()
        self.file_name = '{0}/{1}'.format(self.temp_dir, self.name)
        os.mkfifo(self.file_name)

    def __exit__(self, exception_type, exception_value, exception_traceback):
        os.remove(self.file_name)
        os.rmdir(self.temp_dir)

class PairedTemporaryFifos(object):
    def __init__(self, name='fifo'):
        self.name = name

    def __enter__(self):
        self.temp_dir = tempfile.mkdtemp()
        
        # For peace of mind, make sure there are no %'s in the directory name
        # because bowtie2 will be doing a string replacement on %'s.
        while '%' in self.temp_dir:
            os.rmdir(self.temp_dir)
            self.temp_dir = tempfile.mkdtemp()
        
        self.file_name_template = '{0}/R%_{1}.fastq'.format(self.temp_dir, self.name)

        self.R1_file_name = self.file_name_template.replace('%', '1')
        os.mkfifo(self.R1_file_name)

        self.R2_file_name = self.file_name_template.replace('%', '2')
        os.mkfifo(self.R2_file_name)

    def __exit__(self, exception_type, exception_value, exception_traceback):
        os.remove(self.R1_file_name)
        os.remove(self.R2_file_name)
        os.rmdir(self.temp_dir)

class ThreadFastqWriter(threading.Thread):
    def __init__(self, reads, file_name):
        threading.Thread.__init__(self)
        self.daemon = True
        self.reads = reads
        self.file_name = file_name
        self.start()

    def run(self):
        with open(self.file_name, 'w') as fifo_fh:
            for read in self.reads:
                fifo_fh.write(str(read))

class ThreadPairedFastqWriter(threading.Thread):
    def __init__(self, read_pairs, R1_fn, R2_fn):
        threading.Thread.__init__(self)
        self.daemon = True
        self.read_pairs = read_pairs
        self.R1_fn = R1_fn
        self.R2_fn = R2_fn
        self.start()

    def run(self):
        with open(self.R1_fn, 'w') as R1_fh, open(self.R2_fn, 'w') as R2_fh:
            for R1, R2 in self.read_pairs:
                R1_fh.write(str(R1))
                R2_fh.write(str(R2))

def confirm_index_exists(index_prefix):
    with open(os.devnull, 'w') as devnull:
        try:
            command = ['bowtie2-inspect', '--names', index_prefix]
            subprocess.check_call(command, stdout=devnull, stderr=devnull)
        except subprocess.CalledProcessError:
            raise ValueError('Index prefix {0} does not exist'.format(index_prefix))

def launch_bowtie2(index_prefix,
                   R1_fn,
                   R2_fn,
                   output_file_name,
                   error_file_name,
                   bam_output,
                   by_name,
                   custom_binary,
                   **options):
    kwarg_to_bowtie2_argument = [
        ('aligned_reads_file_name',   ['--al', options.get('aligned_reads_file_name')]),
        ('unaligned_reads_file_name', ['--un', options.get('unaligned_reads_file_name')]),
        ('aligned_pairs_file_name',   ['--al-conc', options.get('aligned_pairs_file_name')]),
        ('unaligned_pairs_file_name', ['--un-conc', options.get('unaligned_pairs_file_name')]),
        ('omit_secondary_seq',        ['--omit-sec-seq']),
        ('memory_mapped_IO',          ['--mm']),
        ('local',                     ['--local']),
        ('ignore_quals',              ['--ignore-quals']),
        ('threads',                   ['--threads', str(options.get('threads')), '--reorder']),
        ('seed_length',               ['-L', str(options.get('seed_length'))]),
        ('seed_failures',             ['-D', str(options.get('seed_failures'))]),
        ('reseed',                    ['-R', str(options.get('reseed'))]),
        ('seed_mismatches',           ['-N', str(options.get('seed_mismatches'))]),
        ('seed_interval_function',    ['-i', options.get('seed_interval_function')]),
        ('gbar',                      ['--gbar', str(options.get('gbar'))]),
        ('report_all',                ['-a']),
        ('report_up_to',              ['-k', str(options.get('report_up_to'))]),
        ('fasta_input',               ['-f']),
        ('report_timing',             ['-t']),
        ('omit_unaligned',            ['--no-unal']),
        ('min_insert_size',           ['-I', str(options.get('min_insert_size'))]),
        ('max_insert_size',           ['-X', str(options.get('max_insert_size'))]),
        ('forward_forward',           ['--ff']),
        ('score_min',                 ['--score-min', options.get('score_min')]),
        ('maximum_ambiguous',         ['--n-ceil', options.get('maximum_ambiguous')]),
        ('ambiguous_penalty',         ['--np', str(options.get('ambiguous_penalty'))]),
        ('allow_dovetail',            ['--dovetail']),
        ('no_mixed',                  ['--no-mixed']),
    ]

    if custom_binary:
        bowtie2_command = ['/home/jah/.local/src/bowtie2-dev/bowtie2']
    else:
        bowtie2_command = ['bowtie2']

    confirm_index_exists(index_prefix)
    
    for kwarg, bowtie2_argument in kwarg_to_bowtie2_argument:
        if kwarg in options:
            value = options.pop(kwarg)
            # If set to false, don't add the argument.
            if value:
                bowtie2_command.extend(bowtie2_argument)

    # Options are getting popped, so if anything is left, then it was
    # something that wasn't being looked for.
    if len(options) > 0:
        raise ValueError('Unknown keyword argument', options)

    bowtie2_command.extend(['-x', index_prefix])

    if R2_fn != None:
        bowtie2_command.extend(['-1', R1_fn])
        bowtie2_command.extend(['-2', R2_fn])
    else:
        bowtie2_command.extend(['-U', R1_fn])

    error_file = open(error_file_name, 'w')
    if not bam_output:
        bowtie2_command.extend(['-S', output_file_name])

        bowtie2_process = subprocess.Popen(bowtie2_command,
                                           stderr=error_file,
                                          )
        process_to_return = bowtie2_process
    else:
        bowtie2_process = subprocess.Popen(bowtie2_command,
                                           stdout=subprocess.PIPE,
                                           stderr=error_file,
                                          )
        view_command = ['samtools', 'view', '-ubh', '-']
        sort_command = ['samtools', 'sort',
                        '-T', output_file_name,
                        '-o', output_file_name,
                       ]
        if by_name:
            sort_command.append('-n')

        sort_command.append('-')

        view_process = subprocess.Popen(view_command,
                                        stdin=bowtie2_process.stdout,
                                        stdout=subprocess.PIPE,
                                       )
        sort_process = subprocess.Popen(sort_command,
                                        stdin=view_process.stdout,
                                        stderr=subprocess.PIPE,
                                       )
        bowtie2_process.stdout.close()
        view_process.stdout.close()
        process_to_return = sort_process

    return process_to_return, bowtie2_command

def _map_bowtie2(index_prefix,
                 R1_fn,
                 R2_fn,
                 output_file_name,
                 error_file_name='/dev/null',
                 custom_binary=False,
                 bam_output=False,
                 by_name=False,
                 reads=None,
                 read_pairs=None,
                 yield_mappings=False,
                 yield_unaligned=False,
                 **options):
    using_input_fifos = reads != None or read_pairs != None
    is_paired = R2_fn != None or read_pairs != None

    if reads:
        input_fifo_source = TemporaryFifo(name='input_fifo.fastq')
    elif read_pairs:
        input_fifo_source = PairedTemporaryFifos(name='input')
    else:
        input_fifo_source = DoNothing()

    if yield_unaligned:
        if is_paired:
            output_fifo_source = PairedTemporaryFifos(name='unaligned')
        else:
            output_fifo_source = TemporaryFifo(name='unaligned.fastq')
    elif yield_mappings:
        output_fifo_source = TemporaryFifo(name='output_fifo.sam')
    else:
        output_fifo_source = DoNothing()
    
    with input_fifo_source, output_fifo_source:
        if reads:
            R1_fn = input_fifo_source.file_name
            writer = ThreadFastqWriter(reads, R1_fn)
        elif read_pairs:
            R1_fn = input_fifo_source.R1_file_name
            R2_fn = input_fifo_source.R2_file_name
            writer = ThreadPairedFastqWriter(read_pairs, R1_fn, R2_fn)

        if yield_unaligned:
            if is_paired:
                unal_template_fn = output_fifo_source.file_name_template
                unal_R1_fn = output_fifo_source.R1_file_name
                unal_R2_fn = output_fifo_source.R2_file_name
                options['unaligned_pairs_file_name'] = unal_template_fn
            else:
                unal_R1_fn = output_fifo_source.file_name
                options['unaligned_reads_file_name'] = unal_R1_fn
        elif yield_mappings:
            output_file_name = output_fifo_source.file_name

        bowtie2_process, bowtie2_command = launch_bowtie2(index_prefix,
                                                          R1_fn,
                                                          R2_fn,
                                                          output_file_name,
                                                          error_file_name,
                                                          bam_output,
                                                          by_name,
                                                          custom_binary,
                                                          **options)

        if yield_unaligned:
            if is_paired:
                for read_pair in fastq.read_pairs(unal_R1_fn, unal_R2_fn):
                    yield read_pair
            else:
                for read in fastq.reads(unal_R1_fn):
                    yield read
        elif yield_mappings:
            sam_file = pysam.AlignmentFile(str(output_file_name), 'r')
            yield sam_file
            for read in sam_file:
                yield read

        _, err_output = bowtie2_process.communicate()
        if bowtie2_process.returncode != 0:
            raise subprocess.CalledProcessError(bowtie2_process.returncode,
                                                bowtie2_command,
                                                err_output,
                                               )
        if bam_output and not by_name:
            pysam.index(str(output_file_name))

def map_bowtie2(index_prefix,
                R1_fn=None,
                R2_fn=None,
                output_file_name=None,
                bam_output=False,
                by_name=False,
                error_file_name='/dev/null',
                custom_binary=False,
                reads=None,
                read_pairs=None,
                yield_mappings=False,
                yield_unaligned=False,
                **options):

    if reads and read_pairs:
        raise RuntimeError('Can\'t give unpaired_Reads and paired_Reads')

    if yield_unaligned and yield_mappings:
        raise RuntimeError('Can\'t yield unaligned and mappings.')

    if output_file_name == None and yield_mappings == False:
        raise RuntimeError('Need to give output_file_name or yield_mappings')
    
    if read_pairs:
        # Can't used named pipes for paired Reads without custom binary because
        # of buffer size mismatch.
        custom_binary = True

    generator = _map_bowtie2(index_prefix,
                             R1_fn=R1_fn,
                             R2_fn=R2_fn,
                             output_file_name=output_file_name,
                             error_file_name=error_file_name,
                             custom_binary=custom_binary,
                             bam_output=bam_output,
                             by_name=by_name,
                             reads=reads,
                             read_pairs=read_pairs,
                             yield_mappings=yield_mappings,
                             yield_unaligned=yield_unaligned,
                             **options)
    if yield_unaligned:
        return generator
    elif yield_mappings:
        sam_file = next(generator)
        return sam_file, generator
    else:
        # There isn't a real yield in generator, so calling next() is just
        # executing the function.
        try:
            next(generator)
        except StopIteration:
            pass

def map_tophat(reads_file_names,
               bowtie2_index,
               tophat_dir,
               gtf_file_name=None,
               transcriptome_index=None,
               num_threads=1,
               no_sort=False,
               keep_temporary_files=True,
              ):

    joined_reads_names = ','.join(str(fn) for fn in reads_file_names)

    options = [
        '--no-novel-juncs',
        '--num-threads', str(num_threads),
        '--output-dir', str(tophat_dir),
        '--report-secondary-alignments',
        '--read-realign-edit-dist', '0',
        #'--read-mismatches', '6',
        #'--read-edit-dist', '6',
        #'--b2-n-ceil', 'C,10,0',
        #'--b2-L', '10',
    ]

    if gtf_file_name is not None:
        options.extend(['--GTF', str(gtf_file_name)])
    if transcriptome_index is not None:
        options.extend(['--transcriptome-index', str(transcriptome_index)])
    if no_sort:
        options.append('--no-sort-bam')
    if keep_temporary_files:
        options.append('--keep-tmp')

    tophat_command = ['tophat2'] + options + [str(bowtie2_index), joined_reads_names]

    # tophat maintains its own logs of everything that is written to the
    # console, so discard output.
    try:
        output = subprocess.check_output(tophat_command, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        print('tophat command returned code {0}'.format(e.returncode))
        print('full command was:\n\t{0}'.format(' '.join(tophat_command)))
        print('output from tophat was:\n\t{0}'.format(e.output))
        raise ValueError

    # If there were no unmapped reads, tophat won't create a file. I want to be
    # able to assume that one exists.
    accepted_hits_fn = '{0}/accepted_hits.bam'.format(tophat_dir)
    unmapped_fn = '{0}/unmapped.bam'.format(tophat_dir)
    if not os.path.exists(unmapped_fn):
        template = pysam.AlignmentFile(accepted_hits_fn)
        empty_unmapped = pysam.AlignmentFile(unmapped_fn, 'wb', template=template)
        template.close()
        empty_unmapped.close()

    if not no_sort:
        pysam.index(accepted_hits_fn)

def map_tophat_paired(R1_fn,
                      R2_fn,
                      bowtie2_index,
                      gtf_file_name,
                      transcriptome_index,
                      tophat_dir,
                      num_threads=1,
                      no_sort=False,
                     ):
    options = [
        '--GTF', gtf_file_name,
        '--no-novel-juncs',
        '--num-threads', str(num_threads),
        '--output-dir', tophat_dir,
        '--transcriptome-index', transcriptome_index,
        '--report-secondary-alignments',
        '--read-realign-edit-dist', '0',
    ]
    if no_sort:
        options.append('--no-sort-bam')

    tophat_command = ['tophat2'] + options + [bowtie2_index, R1_fn, R2_fn]

    # tophat maintains its own logs of everything that is written to the
    # console, so discard output.
    subprocess.check_output(tophat_command, stderr=subprocess.STDOUT)
    
    accepted_hits_fn = '{0}/accepted_hits.bam'.format(tophat_dir)
    if not no_sort:
        pysam.index(accepted_hits_fn)

def map_STAR(R1_fn, index_dir, output_prefix,
             R2_fn=None,
             num_threads=1,
             sort=True,
             include_unmapped=False,
             bam_fn=None,
            ):
    if sort:
        bam_suffix = 'Aligned.sortedByCoord.out.bam'
        sort_option = 'SortedByCoordinate'
    else:
        bam_suffix = 'Aligned.out.bam'
        sort_option = 'Unsorted'

    if include_unmapped:
        unmapped_option = 'Within'
    else:
        unmapped_option = 'None'

    STAR_command = [
        'STAR',
        '--genomeDir', index_dir,
        '--outSAMtype', 'BAM', sort_option,
        '--outSAMunmapped', unmapped_option,
        '--outSAMattributes', 'MD',
        '--limitBAMsortRAM', '1345513406',
        '--outFilterScoreMinOverLread', '0.2',
        '--outFilterMatchNminOverLread', '0.2',
        '--alignIntronMax', '1',
        '--runThreadN', str(num_threads),
        '--outFileNamePrefix', output_prefix,
        '--readFilesIn', R1_fn,
    ]
    if R2_fn is not None:
        STAR_command.append(R2_fn)

    subprocess.check_output(STAR_command)

    initial_bam_fn = str(output_prefix) + bam_suffix

    if bam_fn is None:
        bam_fn = initial_bam_fn
    else:
        shutil.move(initial_bam_fn, bam_fn)

    if sort:
        pysam.index(str(bam_fn))

def build_STAR_index(fasta_files, index_dir, wonky_param=None):
    total_length = 0
    for fasta_file in fasta_files:
        for name, entry in genomes.parse_fai(fasta_file + '.fai').items():
            total_length += entry.length

    if wonky_param is None:
        wonky_param = int(min(14, np.log2(total_length) / 2. - 1))

    STAR_command = [
        'STAR',
        '--runMode', 'genomeGenerate',
        '--genomeDir', index_dir,
        '--genomeFastaFiles', ' '.join(fasta_files),
        '--genomeSAindexNbases', str(wonky_param),
    ]
    subprocess.check_output(STAR_command)

def _map_STAR(reads,
              index_prefix,
              output_prefix,
             ):
    input_fifo_source = TemporaryFifo(name='input_fifo.fastq')

    with input_fifo_source:
        R1_fn = input_fifo_source.file_name
        writer = ThreadFastqWriter(reads, R1_fn)

        STAR_process, STAR_command = map_STAR(R1_fn,
                                              index_prefix,
                                              output_prefix,
                                             )

        _, err_output = STAR_process.communicate()
        if STAR_process.returncode != 0:
            raise subprocess.CalledProcessError(STAR_process.returncode,
                                                STAR_command,
                                                err_output,
                                               )
