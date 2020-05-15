module COCO

using DataDeps
using JSON3
using JuliaDB
using Images
using LearnBase
using PoseEstimation: Joint, PoseConfig
using StaticArrays

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
    register(DataDep(
        "coco_keypoint_annotations",
        COCO_COPYRIGHT,
        "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
        "113a836d90195ee1f884e704da6304dfaaecff1f023f49b6ca93c4aaae470268",
        post_fetch_method = path -> (unpack(path); preparecoco(dirname(path)))
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
        (16, 14),
        (14, 12),
        (17, 15),
        (15, 13),
        (12, 13),
        (6, 12),
        (7, 13),
        (6, 7),
        (6, 8),
        (7, 9),
        (8, 10),
        (9, 11),
        (2, 3),
        (1, 2),
        (1, 3),
        (2, 4),
        (3, 5),
        (4, 6),
        (5, 7)
    ]
)

const MEANS = SVector(0.46339586534149074, 0.44807951561830006, 0.41191946181562883)
const STDS = SVector(0.2353513819415526, 0.2328964398675012, 0.23061464879094382)


struct COCOKeypoints
    imagefolder
    t::IndexedTable
    COCOKeypoints(imagefolder) = new(
        imagefolder,
        JuliaDB.load(joinpath(datadep"coco_keypoint_annotations", "coco.jdb")))
end


function LearnBase.getobs(ds::COCOKeypoints, idx)
    annotation = ds.t[idx]
    image = Images.load(
        getimagefile(ds.imagefolder, annotation.image_id, annotation.isvalid))

    return (image = image, pose = annotation.keypoints, config = CONFIG)
end

LearnBase.nobs(ds::COCOKeypoints) = length(ds.t)


getimagefile(folder, image_id, isvalid) = joinpath(
    folder,
    isvalid ? "val2017" : "train2017",
    "$(lpad(image_id, 12, "0")).jpg"
)

# Dataset preparation

function preparecoco(annotationsfolder)
    trainannspath = joinpath(annotationsfolder, "annotations/person_keypoints_train2017.json")
    validannspath = joinpath(annotationsfolder, "annotations/person_keypoints_val2017.json")

    trainannot = JSON3.read(read(trainannspath))
    validannot = JSON3.read(read(validannspath))

    t = makecocotable(trainannot.annotations, validannot.annotations)
    save(t, joinpath(annotationsfolder, "coco.jdb"))
    return t
end


function makecocotable(trainanns, validanns)
    traint = annotationtable(trainanns, isvalid = false)
    validt = annotationtable(validanns, isvalid = true)
    return merge(traint, validt)
end

function annotationtable(annotations; isvalid = false)
    columns = [:num_keypoints, :area, :keypoints, :bbox, :id, :image_id]
    data = Dict(s => [ann[s] for ann in annotations] for s in columns)
    data[:num_keypoints] = UInt8.(data[:num_keypoints])
    data[:area] = Float32.(data[:area])
    data[:bbox] = parsebbox.(data[:bbox])
    data[:keypoints] = parsekeypoints.(data[:keypoints])
    data[:isvalid] = fill(isvalid, length(data[:num_keypoints]))

    t = table(data, pkey = :image_id)
    return t
end


function parsebbox(bbox)
    y, x, h, w = bbox
    return [SVector{2,Float32}(y + 1, x + 1), SVector{2,Float32}(y + h, x + w)]
end


function parsekeypoints(keypoints)
    xs = @view keypoints[1:3:end]
    ys = @view keypoints[2:3:end]
    ids = @view keypoints[3:3:end]
    pose = Vector{Union{Nothing, SVector{2, Float32}}}(nothing, length(xs))

    for (k, (x, y, id)) in enumerate(zip(xs, ys, ids))
        if id != 0
            pose[k] = SVector{2,Float32}(y + 1, x + 1)
        end
    end

    return pose
end

end # module
