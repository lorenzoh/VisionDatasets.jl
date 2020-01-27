module VisionDatasets

import LearnBase: getobs, nobs

include("./classification/classification.jl")
include("./classification/ImageNette.jl")
include("./classification/ImageWoof.jl")

include("./poseestimation/poseestimation.jl")
include("./poseestimation/MPII.jl")

export
    # classification
    ClassificationDataset,
    ImageNette,
    ImageWoof,

    # pose estimation
    PoseDataset,
    MPII

end # module
