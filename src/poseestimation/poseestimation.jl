using PoseEstimation: PoseConfig, Joint

struct PoseDataset
    files::AbstractVector{AbstractString}
    poses::AbstractVector{AbstractMatrix{<:Joint}}
    config::PoseConfig
    dir::AbstractString
    sampledata::Dict
    metadata::Dict
    function PoseDataset(imagenames, poses, config, dir, sampledata, metadata)
        length(imagenames) == length(poses) || error("imagenames and poses must be of the same length!")
        isdir(dir) || error("$(dir) could not be opened.")
        return new(imagenames, poses, config, dir, sampledata, metadata)
    end
end

nobs(dataset::PoseDataset) = length(dataset.files)


function getobs(dataset::PoseDataset, idx::Int)
    image = load(joinpath(dataset.dir, dataset.files[idx]))
    poses = dataset.poses[idx]
    return Dict(
        :image => image,
        :poses => poses,
        :config => dataset.config,
        Dict(
            key => dataset.sampledata[key][idx] for key in keys(dataset.sampledata)
        )...
    )
end

function getobs(pd::PoseDataset, idxs::AbstractVector{<:Int})
    return [getobs(pd, idx) for idx in idxs]
end
