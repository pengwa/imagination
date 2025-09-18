1. Stop Docker service

First, stop the Docker service to ensure that no operations are running while you change the configuration.

sudo systemctl stop docker

2. Create the new target folder

Create the folder where you want to store Docker images and containers. For example, let's assume you want to use /mnt/docker:

```
sudo mkdir -p /mnt/docker
```

3. Modify Docker daemon configuration

You need to update the Docker daemon's configuration to point to the new storage location. This can be done by editing the Docker configuration file (/etc/docker/daemon.json).

If the daemon.json file doesn't exist, create it:

```
sudo nano /etc/docker/daemon.json
```

Add the following content to configure the Docker storage location:

```
{
  "data-root": "/mnt/docker"
}
```

This tells Docker to store all images, containers, volumes, and other data in the /mnt/docker directory.


4. Restart Docker service

Now, restart the Docker service to apply the changes:

```
sudo systemctl start docker
```

5. Verify the change

Check if the change was applied successfully by running:

```
docker info | grep "Docker Root Dir"
```

This should now show the new path (/mnt/docker).
