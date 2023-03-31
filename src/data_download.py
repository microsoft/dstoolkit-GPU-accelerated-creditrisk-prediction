import os
import tarfile
import time
import urllib.request


def extract_files(fname, extraction_dir):
    tar = tarfile.open(fname, "r:gz")
    tar.extractall(extraction_dir)
    tar.close()
    print(f"Successfully extracted contents of '{fname}' to '{extraction_dir}'")
    print(f"Contents in directory '{extraction_dir}': {os.listdir(extraction_dir)}")

    print("Directory Structure:")
    print ('--------------------------------')
    print()
    for (root,dirs,files) in os.walk(extraction_dir, topdown=True):
        print (root)
        print (dirs)
        print (files)
        print ('--------------------------------')

def download_from_url(url, dir_):
    if not os.path.exists(dir_):
        os.makedirs(dir_)
    fname = url.split("/")[-1]
    fname = os.path.join(dir_, fname)

    if not os.path.exists(fname):
        max_attempts = 80
        attempts = 0
        sleeptime = 10
        while attempts < max_attempts:
            time.sleep(sleeptime)
            try:
                response = urllib.request.urlopen(url, timeout = 5)
                content = response.read()
                f = open(fname, 'wb' )
                f.write( content )
                f.close()
                print("Successfully downloaded data!")
                break
            except Exception as e:
                attempts += 1
                print(type(e))
                print(e)
    return fname
    
def download_and_extract_data(url, dir_):
    fname = download_from_url(url, dir_)    
    extract_files(fname, dir_)