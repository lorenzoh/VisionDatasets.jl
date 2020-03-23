module MPII

using DataDeps
using JSON3
using PoseEstimation: Joint, PoseConfig

using ..VisionDatasets: PoseDataset, groupby


function __init__()
    register(DataDep(
        "mpii_annotations",
        """
        MPII dataset as published on "http://human-pose.mpi-inf.mpg.de/#download"

        LICENSE
        -------
        Copyright (c) 2015, Max Planck Institute for Informatics
        All rights reserved.

        Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

        1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

        2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

        THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
        """,
        [
            "https://gist.github.com/lorenzoh/7579e9493775ba2dd4018c7cf44aa112/raw/0b7c4df4d736bc7bcb349abea57edc66b6baa936/mpiivalid.json",
            "https://gist.github.com/lorenzoh/ed17568b9a5440b60e19da6e5efabf73/raw/aa2a7afefa4672429e2cafb0d219caed4a27bce1/mpiitrain.json"
        ],
        "604bc9286027c072a42620d81e253959dbb585a8b84bf246a98905d583075e39";
    ))
end

const CONFIG = PoseConfig(
    16,
    [
        "r ankle",
        "r knee",
        "r hip",
        "l hip",
        "l knee",
        "l ankle",
        "pelvis",
        "thorax",
        "upper neck",
        "head top",
        "r wrist",
        "r elbow",
        "r shoulder",
        "l shoulder",
        "l elbow",
        "l wrist",
    ],
    [
        (3, 2),
        (15, 16),
        (12, 11),
        (8, 7),
        (9, 10),
        (9, 8),
        (5, 6),
        (7, 4),
        (7, 3),
        (4, 5),
        (9, 14),
        (9, 13),
        (14, 15),
        (13, 12),
        (2, 1)
    ]
)


function mpii(imgpath = datadep"mpii_images", annpath = datadep"mpii_annotations")
    trainanns = open(JSON3.read, joinpath(annpath, "mpiitrain.json"))
    validanns = open(JSON3.read, joinpath(annpath, "mpiivalid.json"))

    traindict = groupby(trainanns, :image)
    validdict = groupby(validanns, :image)

    splitdict = Dict()
    for (d, split) in zip((traindict, validdict), (:train, :valid))
        for key in keys(d)
            splitdict[key] = split
        end
    end

    anns = merge(traindict, validdict)

    return PoseDataset(
        [path for (path, _) in anns],
        [parseannotation(ann) for (_, ann) in anns],
        CONFIG,
        imgpath,
        Dict(),
        Dict(
            :split => [splitdict[path] for (path, _) in anns],
            :stats => Dict(
                :means => [0.46339586534149074, 0.44807951561830006, 0.41191946181562883],
                :stds => [0.2353513819415526, 0.2328964398675012, 0.23061464879094382],
            ),
        )
    )
end



function parsejoint(j)::Union{Nothing, <:Tuple}
    x, y = j
    return (y, x) == (-1, -1) ? nothing : (y+1, x+1)
end

function parseannotation(ann)::AbstractMatrix{Joint}
    vcat([reshape(parsejoint.(p.joints), 1, :) for p in ann]...)
end

export mpii


end # module
