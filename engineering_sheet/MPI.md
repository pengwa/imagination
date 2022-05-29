https://www.open-mpi.org/faq/?category=running

    34. What MPI environmental variables exist?
    Beginning with the v1.3 release, Open MPI provides the following environmental variables that will be defined on every MPI process:

    OMPI_COMM_WORLD_SIZE - the number of processes in this process's MPI_COMM_WORLD
    OMPI_COMM_WORLD_RANK - the MPI rank of this process in MPI_COMM_WORLD
    OMPI_COMM_WORLD_LOCAL_RANK - the relative rank of this process on this node within its job. For example, if four processes in a job share a node, they will each be given a local rank ranging from 0 to 3.
    OMPI_UNIVERSE_SIZE - the number of process slots allocated to this job. Note that this may be different than the number of processes in the job.
    OMPI_COMM_WORLD_LOCAL_SIZE - the number of ranks from this job that are running on this node.
    OMPI_COMM_WORLD_NODE_RANK - the relative rank of this process on this node looking across ALL jobs.
    
 
