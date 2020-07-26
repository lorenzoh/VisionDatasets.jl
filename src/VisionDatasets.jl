module VisionDatasets

import LearnBase: getobs, nobs

include("./classification/classification.jl")
include("./classification/ImageNette.jl")
include("./classification/ImageWoof.jl")

include("./poseestimation/poseestimation.jl")
include("./poseestimation/utils.jl")
include("./poseestimation/COCO.jl")
include("./poseestimation/MPII.jl")
include("./poseestimation/MPI_INF_3DHP.jl")
include("./poseestimation/MPI_3DPW.jl")

export
    getobs,
    nobs,

    # classification
    ClassificationDataset,
    ImageNette,
    ImageWoof,

    # pose estimation
    PoseDataset,
    COCO,
    MPII,
    MPI_INF_3DHP,
    MPI_3DPW

end # module
