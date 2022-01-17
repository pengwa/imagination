ignore_git_permission:

  git config core.fileMode false


reset submodules recursively:

  git submodule foreach --recursive git reset --hard
