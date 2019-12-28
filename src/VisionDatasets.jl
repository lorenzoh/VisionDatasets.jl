module VisionDatasets

include("./classification/classification.jl")
include("./classification/ImageNette.jl")
include("./classification/ImageWoof.jl")

export ClassificationDataset, ImageNette, ImageWoof

end # module
