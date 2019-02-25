try:
    import subprocess32 as subprocess
except ImportError:
    import subprocess
import os
import contextlib
import sequencing.Parallel
import yaml
import sys

def piece(srr_fn, num_pieces, which_piece, paired=False):
    root, _ = os.path.splitext(srr_fn)
    yaml_fn = '{0}.yaml'.format(root)

    if not os.path.exists(yaml_fn):
        raise ValueError('yaml doesn\'t exist for {0}'.format(srr_fn))

    with open(yaml_fn) as yaml_fh:
        info = yaml.load(yaml_fh)

    total_spots = info['total_spots']

    bounds = sequencing.Parallel.get_bounds(total_spots, num_pieces)
    first = bounds[which_piece] + 1
    last = bounds[which_piece + 1]

    with dump_spots(srr_fn, first, last, paired) as lines:
        for line in lines:
            # Note: empirically, these lines are bytes objects.
            yield line.decode()

@contextlib.contextmanager
def dump_spots(srr_fn, first, last, paired):
    if paired:
        name_format = '@$ac.$si.$ri'
    else:
        name_format = '@$ac.$si'

    command = [
        'fastq-dump',
        '--dumpbase',
        '--minSpotId', str(first),
        '--maxSpotId', str(last),
        '--defline-seq', name_format,
        '--stdout',
        srr_fn,
    ]

    if paired:
        command.insert(1, '--split-spot')

    process = subprocess.Popen(command,
                               stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE,
                              )
    lines = iter(process.stdout)
    try:
        yield lines
    finally:
        process.terminate()
        process.stdout.close()
        for line in process.stderr:
            if not (line.startswith(b'Read') or line.startswith(b'Written')):
                sys.stderr.write(line)
