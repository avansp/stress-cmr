Some changes from the original repo:

1. I'm using [poetry](https://python-poetry.org/) for managing packages & environment.

2. This is mainly used for command line application and I'm using [typer](https://typer.tiangolo.com/) to create those apps.

3. Use the docker folder to build, run and interactively use a docker container. 

## Dockering

### Building a new image
```bash
$ docker build -t stress-cmr:latest . 
```

### Start the docker container

```
$ ./start_docker.sh [CODE_DIR] [DATA_DIR] [IMAGE_NAME]
```
where
* `CODE_DIR` is where the repo is. 
* `DATA_DIR` is a data folder where `final.csv` is located. This is also the folder where training, testing & validation results are saved.
* `IMAGE_NAME` the docker image name when you build this.

E.g.
```
$ ./start_docker.sh .. ../../data stress-cmr:latest
```

The container will run interactively using the current user id and will be detached. The container name is `stress-cmr` 

```
$ docker container ls
CONTAINER ID   IMAGE               COMMAND   CREATED          STATUS          PORTS     NAMES
e1a9781cc5af   stress-cmr:latest   "bash"    47 seconds ago   Up 46 seconds             stress-cmr
```

### Log into the container
```
$ ./enter_docker.sh
I have no name!@e1a9781cc5af:~$
```

### Log as root
```
$ ./sudo_docker.sh
root@e1a9781cc5af:~#
```