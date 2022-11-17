### ignore_git_permission:

    git config core.fileMode false

### add safe directory

    git config --global --add safe.directory '*'


### reset submodules recursively:

    git submodule foreach --recursive git reset --hard



\\codeflow\public\cf.cmd openGitHubPr -webUrl https://github.com/microsoft/onnxruntime/pull/
