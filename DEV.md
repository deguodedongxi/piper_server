# Building Project

```make``` - Run this command in the parent directory to build the project.

# Running Project


```CMD
cd build

./piper_server --port 8080

``` 
> Run this command in the `build` directory to run the project.

# Creating new package versions

> Make sure what the current tag version of the project is. You can check this by running the following command:

```CMD

git tag

```

> Creating a new tag and pushing it will create a new release on the github repository. 

```CMD 

git tag v1.0

git push origin v1.0

git push --tags

```