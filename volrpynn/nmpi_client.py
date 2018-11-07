#!/usr/bin/env python3

import nmpi
import logging
import time
import os
import sys
import urllib.request
import re
from contextlib import contextmanager

@contextmanager
def silence_stdout():
    new_target = open(os.devnull, "w")
    old_target, sys.stdout = sys.stdout, new_target
    try:
        yield new_target
    finally:
        sys.stdout = old_target

@contextmanager
def silence_stderr():
    new_target = open(os.devnull, "w")
    old_target, sys.stderr = sys.stderr, new_target
    try:
        yield new_target
    finally:
        sys.stderr = old_target
        
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("NMPI")
logger.propagate = False

logger = logging.getLogger("nm_client")

if __name__ == "__main__":
    import argparse    
    parser = argparse.ArgumentParser()
    parser.add_argument("--username", type=str, default = os.environ.get('HBP_USERNAME', None))
    parser.add_argument("--password", type=str, default = os.environ.get('HBP_PASSWORD', None)) 
    parser.add_argument("--collab", type=str, default = os.environ.get('HBP_COLLAB', None))
    parser.add_argument("--software_version", type=str, default = 'nmpm_software/2017-09-11-spack-2017-01-26-1')
    parser.add_argument("--script", type=str)
    parser.add_argument("--output", type=str, default = ".")
    parser.add_argument("--wafer", type=int, default = 21)
    parser.add_argument("--hicann", type=int, default = 320)
    parser.add_argument("--command", type=str, default = None)
    parser.add_argument("--baese_url", type=str, default = 'https://brainscales-r.kip.uni-heidelberg.de:7443/')
    parser.add_argument('-i', '--inputs', nargs='+', type=str)    
    args = parser.parse_args()

    
    client = nmpi.Client(username = args.username, password = args.password)
    collab_id = client.my_collabs()[args.collab]["id"]
    logger.info("Collab ID: {}".format(collab_id))
    hw_config = {'WAFER_MODULE': args.wafer, 'HICANN': args.hicann, 'SOFTWARE_VERSION': args.software_version}
    logger.info("Hardware config: {}".format(hw_config))

    job_path = ''
    command = ''
    if args.command is None:
        command = "run.py --wafer {} --hicann {}".format(args.wafer, args.hicann)
    else:
        command = args.command
    with silence_stdout():
        job_path = client.submit_job(source=args.script, 
                                    platform=nmpi.BRAINSCALES, 
                                    collab_id=collab_id,
                                    inputs=["net.json"], # TODO: Hardcoded for now.
                                    config=hw_config, 
                                    command=command)
    
    # Each job gets assigned a id, which can be used to track its progress and
    # fetch any results it produces.

    job_id = job_path.split('/')[-1]
    logger.info("Job {} submitted".format(job_id))
    state = 'submitted'
    with silence_stderr():
        while state in ['submitted', 'running']:
            next_state = client.job_status(job_id)
            if (next_state != state):
                logger.info("Job {} state changed to {}.".format(job_id, next_state))
            time.sleep(2) # TODO: There might be a better way than polling.
            state = next_state

    save_dir = args.output
    data_folder = os.path.join(save_dir, 'job_{}/'.format(job_id))
    os.makedirs(data_folder)
    
    # After a job has finished, all results are stored in a *publicly accessible*
    # server directory and can be retrieved. Here we just use the fact that the
    # Apache server has a default directory listing page, which can be used
    # to discover all files that have been generated. 
    #
    # In principle we know all files that will be generated, so we could fetch them from
    #
    # ${base_url}/nmpi/job_${job_id}/$filename
    #
    # directly. Here we opt for the lazy method of relying on the specifics of the
    # Apache directory listing instead.

    pattern = '<a href=".*?">(.*?)</a>'
    url = args.base_url + 'nmpi/job_{}/'.format(job_id)
    logger.info("Retrieving results into {}".format(data_folder))
    response = urllib.request.urlopen(url).read().decode('utf-8')
    
    for filename in re.findall(pattern, response)[5:]:
        fp = open(os.path.join(data_folder, filename), 'w')
        logger.info("Downloading into {}".format(os.path.join(data_folder, filename)))
        fp.write(urllib.request.urlopen(url + filename).read().decode('utf-8'))
        fp.close()
