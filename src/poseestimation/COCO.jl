module COCO

using DataDeps
using JSON3
using PoseEstimation: Joint, PoseConfig

using ..VisionDatasets: PoseDataset, groupby

const COCO_COPYRIGHT = """
Terms of Use
Annotations & Website
The annotations in this dataset along with this website belong to the COCO Consortium and are licensed under a Creative Commons Attribution 4.0 License.


Images
The COCO Consortium does not own the copyright of the images. Use of the images must abide by the Flickr Terms of Use. The users of the images accept full responsibility for the use of the dataset, including but not limited to the use of any copies of copyrighted images that they may create from the dataset.

Software
Copyright (c) 2015, COCO Consortium. All rights reserved. Redistribution and use software in source and binary form, with or without modification, are permitted provided that the following conditions are met:

Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
Neither the name of the COCO Consortium nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
THIS SOFTWARE AND ANNOTATIONS ARE PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""


function __init__()

    # Register keypointannotations valid and train
    register(DataDep(
        "coco_keypoint_annotations",
        COCO_COPYRIGHT,
        "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
        "113a836d90195ee1f884e704da6304dfaaecff1f023f49b6ca93c4aaae470268",
        post_fetch_method = unpack
    ))



end


const CONFIG = PoseConfig(
    17,
    [
        "nose","left_eye","right_eye","left_ear","right_ear",
        "left_shoulder","right_shoulder","left_elbow","right_elbow",
        "left_wrist","right_wrist","left_hip","right_hip",
        "left_knee","right_knee","left_ankle","right_ankle"
    ],
    [
        (16,14),
        (14,12),
        (17,15),
        (15,13),
        (12,13),
        (6,12),
        (7,13),
        (6,7),
        (6,8),
        (7,9),
        (8,10),
        (9,11),
        (2,3),
        (1,2),
        (1,3),
        (2,4),
        (3,5),
        (4,6),
        (5,7)
    ]
)

function coco_keypoints(imagespath, annotationpath = datadep"coco_keypoint_annotations")
    trainannspath = joinpath(annotationpath, "annotations/person_keypoints_train2017.json")
    validannspath = joinpath(annotationpath, "annotations/person_keypoints_val2017.json")

    trainanns = JSON3.read(read(trainannspath))
    validanns = JSON3.read(read(validannspath))

    imagefiledict = merge(
        Dict(image.id => "train2017/$(image.file_name)" for image in trainanns.images),
        Dict(image.id => "val2017/$(image.file_name)" for image in validanns.images)
    )

    traindict = groupby(trainanns.annotations, :image_id)
    validdict = groupby(validanns.annotations, :image_id)

    splitdict = Dict()
    for (d, split) in zip((traindict, validdict), (:train, :valid))
        for key in keys(d)
            splitdict[key] = split
        end
    end

    anns = merge(traindict, validdict)

    return PoseDataset(
        [imagefiledict[image_id] for (image_id, _) in anns],
        [parseposes(ann) for (_, ann) in anns],
        CONFIG,
        imagespath,
        Dict(
            :missingbboxes => [parsemissingboxes(ann) for (_, ann) in anns]
        ),
        Dict(
            :split => [splitdict[path] for (path, _) in anns],
            :stats => Dict(
                :means => [0.46339586534149074, 0.44807951561830006, 0.41191946181562883],
                :stds => [0.2353513819415526, 0.2328964398675012, 0.23061464879094382],
            ),
        )
    )
end


function parseposes(ann)
    return vcat([reshape(parsekeypoints(a.keypoints), 1, :) for a in ann]...)
end

function parsekeypoints(keypoints)::Vector{Joint}
    xs = keypoints[1:3:end]
    ys = keypoints[2:3:end]
    ids = keypoints[3:3:end];
    return [id == 0 ? nothing : (y+1, x+1) for (x, y, id) in zip(xs, ys, ids)]
end


function parsemissingboxes(ann)::Vector{}
    return boxes = [parsebbox(a.bbox) for a in ann if a.num_keypoints == 0]
end

function parsebbox(cocobbox)::Tuple{Tuple{Float32, Float32}, Tuple{Float32, Float32}}
    x, y, w, h = cocobbox
    return ((y+1, x+1), (y + h, x + w))
end

end # module
