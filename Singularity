Bootstrap: docker
From: python:3.12

%post
    apt install -y libmagickwand-dev

    cd /mnt/path/to/untangle
    python -m pip install --root-user-action=ignore -r requirements.txt
