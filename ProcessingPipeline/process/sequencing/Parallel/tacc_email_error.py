import subprocess
import sys
import smtplib
from email.mime.text import MIMEText
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('job_name', help='name of job')
parser.add_argument('job_dir', help='directory job is run from')
parser.add_argument('job_number', help='job number')

args = parser.parse_args()

try:
    grep_command = ['grep', '-B', '10',
                    'Error',
                    '{job_dir}/{job_name}.e{job_number}'.format(job_dir=args.job_dir,
                                                                job_name=args.job_name,
                                                                job_number=args.job_number,
                                                               )
                   ]
    context = subprocess.check_output(grep_command)
except subprocess.CalledProcessError as e:
    if e.returncode == 1:
        # grep exits with return code 1 on no match
        sys.exit()
    else:
        context = ' '.join(e.cmd)

SMTP_SERVER = 'smtp.gmail.com'
SMTP_PORT = 587

sender = 'taccerrorreport@gmail.com'
password = '329i0D3kH228oap'

session = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)

session.ehlo()
session.starttls()
session.ehlo()
session.login(sender, password)

message = context

subject = 'Possible error in {job_number} - {job_name}'.format(job_number=args.job_number,
                                                               job_name=args.job_name,
                                                              )
recipient = 'jeff.hussmann@gmail.com'
msg = MIMEText(message)
msg['Subject'] = subject
msg['From'] = sender 
msg['To'] = recipient 

session.sendmail(sender, [recipient], msg.as_string())
session.close()
