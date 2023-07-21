import argparse
import os
import subprocess
import time
from datetime import datetime

import moxing as mox

parser = argparse.ArgumentParser(description='s3helper')
parser.add_argument('--job_name', type=str, default='job7')
parser.add_argument('--bucket', type=str, default='bucket-cnsouth1')
parser.add_argument('--data_url', type=str, default='job')
parser.add_argument('--init_method', type=str, default='job')
parser.add_argument('--train_url', type=str, default='job')
cfg = parser.parse_args()

bucket = os.getenv('bucket', cfg.bucket)
# s3_work_dir = f's3://{bucket}//longhui/code/console/'
# local_work_dir = f'/home/ma-user/modelarts/user-job-dir/console/'
# s3_code_dir = f's3://{bucket}/longhui/code/code/unbiased-teacher/'
# local_code_dir = '/home/ma-user/modelarts/user-job-dir/unbiased-teacher'

s3_work_dir = f's3://{bucket}/hushoukang2/queue/console/'
local_work_dir = f'/home/ma-user/modelarts/user-job-dir/queue/console/'
s3_code_dir = f's3://{bucket}/hushoukang2/queue/'
local_code_dir = '/home/ma-user/modelarts/user-job-dir/queue/'

job_name = cfg.job_name
remote_job_file = os.path.join(s3_work_dir, f'{job_name}.sh')
local_job_file = os.path.join(local_work_dir, f'{job_name}.sh')

stop_sign = os.path.join(s3_work_dir, f'{job_name}_stop.sh')


def main():
    print('new helper start @ %s !' % datetime.now())
    tic = time.time()
    while True:
        # 如果有job就跑job
        try:
            if mox.file.exists(remote_job_file):
                try:
                    # 开个进程跑job
                    timestamp = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime())
                    print(f'{timestamp}： new job submitted.')
                    print('copy file(s) from %s to %s ...' % (s3_code_dir, local_code_dir))
                    mox.file.copy_parallel(remote_job_file, local_job_file)
                    mox.file.copy_parallel(s3_code_dir, local_code_dir)
                    proc = subprocess.Popen(f"bash {local_job_file}", shell=True)
                    mox.file.remove(remote_job_file, recursive=False)
                except Exception as e:
                    print(e)
        except:
            time.sleep(5)

        # 偶尔也要歇一歇
        time.sleep(5)

        try:
            if mox.file.exists(stop_sign):
                # job, visited = load_and_remove(stop_sign)
                # if not visited:
                mox.file.remove(stop_sign, recursive=True)
                print('[Info] stop sign detected, battle control terminated!')
                proc.kill()
                os.system("fuser -k /dev/nvidia* ")
                time.sleep(10)
                print("[Info] Process has been killed.")
        except:
            time.sleep(5)


if __name__ == '__main__':
    main()
