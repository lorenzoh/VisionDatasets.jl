module MPI_3DPW

#= TODOs

- move code into module
- project points onto image and check validity
- decide on sample format
- preprocessing
    - normalizing inputs
    - normalizing outputs


=#
using Glob
using CoordinateTransformations
using PoseEstimation
using Images
import LearnBase: getobs, nobs
using PyCall
const pickle = pyimport("pickle")
const py = PyCall.builtin
using StaticArrays

## Constants

CONFIG = PoseConfig(
    24,
    [
        "pelvis",
        "hip_left",
        "hip_right",
        "spine_bottom",
        "knee_left",
        "knee_right",
        "spine_middle",
        "ankle_left",
        "ankle_right",
        "spine_top",
        "toe_left",
        "toe_right",
        "head",
        "clavicle_left",
        "clavicle_right",
        "head_top",
        "shoulder_left",
        "shoulder_right",
        "elbow_left",
        "elbow_right",
        "wrist_left",
        "wrist_right",
        "hand_left",
        "hand_right",
    ],
    [
        (1, 2),
        (1, 3),
        (1, 4),
        (2, 5),
        (3, 6),
        (4, 7),
        (5, 8),
        (6, 9),
        (7, 10),
        (8, 11),
        (9, 12),
        (10, 14),
        (10, 15),
        (10, 13),
        (13, 16),
        (14, 17),
        (15, 18),
        (17, 19),
        (18, 20),
        (19, 21),
        (20, 22),
        (21, 23),
        (22, 24),
    ]
)

# Transformation for 2D coordinate system
FIX2D = LinearMap(SMatrix{2, 2}([0 1; 1 0]))
FIX3D = LinearMap(SMatrix{3, 3}(
    [0  0 -1;
     1  0  0;
     0  -1  0]
    ))


struct MPI3DPWDataset
    sequences
    imagefolder
end

function MPI3DPWDataset(seqfolder::AbstractString, imagefolder::AbstractString)

    return MPI3DPWDataset(
        [loadsequence(path) for path in glob("*.pkl", seqfolder)],
        imagefolder
    )
end


# parsing

"""
    preparesequence(data, actorid)

Load 2D and 3D poses for actor `actorid` from sequence
data `data`. 3D poses are transformed into camera space.
"""
function preparesequence(data, actorid)
    poses3d_world = parse3dposes(data["jointPositions"][actorid])
    Ps = parseprojections(data)

    poses3d = stack([
        map(k -> P.ext * tohom(k) |> fromhom |> FIX3D, pose)
        for (P, pose) in zip(Ps, eachrow(poses3d_world))
    ])
    poses2d = stack([(FIX2D ∘ P).(pose) for (P, pose) in zip(Ps, eachrow(poses3d_world))])
    return poses2d, poses3d
end

struct Projection
    ext
    int
    P
end
Projection(ext, int) = Projection(ext, int, int * ext)
(p::Projection)(x) = fromhom(p.P * tohom(x))

function parseprojections(data)
    extrinsics = [SMatrix{4, 4}(rt) for rt in eachslice(data["cam_poses"], dims = 1)]
    intrinsic = SMatrix{3, 4}(hcat(data["cam_intrinsics"], zeros(3, 1)))
    Ps = map(e -> Projection(e, intrinsic), extrinsics)
    return Ps
end

function parse2dposes(a)
    tmp = reshape(permuteddimsview(a[:,1:2,:], (2, 3, 1)), 36, :)
    permuteddimsview(reinterpret(SVector{2, Float64}, tmp), (2, 1))
end


function parse3dposes(a)
    return permuteddimsview(permuteddimsview(a, (2, 1)) |> _parse3dpose, (2, 1))
end

_parse3dpose(a) = reinterpret(SVector{3, Float64}, a)



# IO

function loadsequence(file)
    pickle = pyimport("pickle")
    return pickle.load(PyCall.builtin.open(file, "rb"), encoding = "bytes")
end


function loadimage(folder, seqname, frame)
    file = joinpath(folder, seqname, "image_$(lpad(string(frame), 5, '0')).jpg")
    return load(file)
end



# Utils

function tohom(x::SVector{3})
    return SVector{4}(x[1], x[2], x[3], 1)
end

function fromhom(X::SVector{3})
    z = X[3]
    return SVector{2}(X[1] / z, X[2] / z)
end


function fromhom(X::SVector{4})
    z = X[4]
    return SVector{3}(X[1] / z, X[2] / z, X[3] / z)
end

stack(xs) = vcat(reshape.(xs, 1, :)...)

end
