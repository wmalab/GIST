
# Create a virtual environment, if you don't already have one
```console
$ conda create env -f environment.yml
```

# Load the new environment
```console
$ conda activate env_viz
 ```

# Install kernel

```
$ python -m ipykernel install --user --name env_viz --display-name env_viz
```

# Submit tunnel job

```
$ sbatch submit_jupyter.sh
```

