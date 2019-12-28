import LearnBase: getobs, nobs
using FileIO: load

struct ClassificationDataset
    dir::AbstractString
    files::AbstractVector{AbstractString}
    labels::AbstractVector{Integer}
    names::AbstractVector{AbstractString}
    data::Dict
end


function getobs(ds::ClassificationDataset, idx::Integer)
    return Dict(
        :image => load(joinpath(ds.dir, ds.files[idx])),
        :label => ds.labels[idx],
        [key => values[idx] for (key, values) in ds.data]...
        )
end
getobs(ds::ClassificationDataset, idxs) = [getobs(ds, idx) for idx in idxs]
nobs(ds::ClassificationDataset) = length(ds.files)


function load_classification_dataset(dir::AbstractString)
    subdirs = scandir(dir)
    dir = length(subdirs) == 1 ? subdirs[1] : dir

    splitdirs = scandir(dir)

    namesset = Set()
    for splitdir in splitdirs
        for name in readdir(splitdir)
            push!(namesset, name)
        end
    end
    names = sort([namesset...])
    classids = 1:length(names)

    name_to_id = Dict(zip(names, classids))

    files = String[]
    ids = Integer[]
    splits = Symbol[]

    for split in readdir(dir)
        for name in readdir(joinpath(dir, split))
            for imagefile in readdir(joinpath(dir, split, name))
                push!(files, joinpath(split, name, imagefile))
                push!(ids, name_to_id[name])
                push!(splits, Symbol(split))
            end
        end
    end
    return ClassificationDataset(dir, files, ids, names, Dict(:split => splits))
end


scandir(dir) = [joinpath(dir, path) for path in readdir(dir)]
