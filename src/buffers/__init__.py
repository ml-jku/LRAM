def make_buffer_class(kind): 
    if kind == "domain":
        from .multi_domain_buffer import MultiDomainTrajectoryReplayBuffer
        return MultiDomainTrajectoryReplayBuffer
    else: 
        from .trajectory_buffer import TrajectoryReplayBuffer
    return TrajectoryReplayBuffer
