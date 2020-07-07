module MPI_INF_3DHP

using Glob: glob
using Images: load
using MAT: matread
using PoseEstimation: PoseConfig
using StaticArrays


const KEYPOINT_NAMES = [
    "spine3", "spine4", "spine2", "spine", "pelvis",
    "neck", "head", "head_top", "left_clavicle", "left_shoulder", "left_elbow",
    "left_wrist", "left_hand",  "right_clavicle", "right_shoulder", "right_elbow", "right_wrist",
    "right_hand", "left_hip", "left_knee", "left_ankle", "left_foot", "left_toe",
    "right_hip" , "right_knee", "right_ankle", "right_foot", "right_toe"
]
const CONFIG = PoseConfig(
    28,
    KEYPOINT_NAMES,
    [
        # head
        (6, 7),
        (7, 8),
        # spine
        (6, 2),
        (2, 1),
        (1, 3),
        (3, 4),
        (4, 5),
        # left arm
        (6, 9),
        (9, 10),
        (10, 11),
        (11, 12),
        (12, 13),
        # right arm
        (6, 14),
        (14, 15),
        (15, 16),
        (16, 17),
        (17, 18),
        # left leg
        (5, 19),
        (19, 20),
        (20, 21),
        (21, 22),
        (22, 23),
        # right leg
        (5, 24),
        (24, 25),
        (25, 26),
        (26, 27),
        (27, 28),
    ]
)



function load_mpi3d(folder; subjects=1:8, sequences = 1:2, cameras=1:14, frames=1:100000)

    data = Dict()

    subject_folders = sort(glob("S[1-8]", folder))
    for (i, subject_folder) in enumerate(subject_folders)
        if !(i in subjects)
            continue
        end
        data[i] = Dict()

        seq_folders = sort(glob("Seq[1-2]", subject_folder))
        for (j, seq_folder) in enumerate(seq_folders)
            if !(j in sequences)
                continue
            end
            data[i][j] = load_sequence(
                seq_folder, i, j;
                cameras = cameras,
                frames = frames
                )
        end
    end

    return data
end


function loadsample(dataset, subject, sequence, camera, frame)
    subdata = dataset[subject][sequence]

    imagedata = subdata[:images][camera]
    imagefile = joinpath(imagedata[:path], imagedata[:files][frame])

    annotationdata = subdata[:annotations][camera]
    pose2d = annotationdata[:annot2d][frame,:] |> parsepose2d
    pose3d = annotationdata[:univannot3d][frame,:] |> parsepose3d

    return Dict(
        :image => load(imagefile),
        :pose2d => pose2d,
        :pose3d => pose3d,
    )


end


function load_sequence(folder, subject_id, sequence_id; cameras=1:14, frames=1:100000)
    files = readdir(folder)
    data = Dict{Symbol, Any}(:path => folder)

    if "imageSequence" in files
        data[:images] = load_videos(joinpath(folder, "imageSequence"), cameras = cameras)
    end

    if "annot.mat" in files
        data[:annotations] = load_annotations(joinpath(folder, "annot.mat"), cameras = cameras)
    end

    return data
end


function load_videos(folder; force = false, cameras = 1:14)
    files = sort(glob("video_*.avi", folder), by=videoordering)

    data = Dict()
    for (i, file) in enumerate(files)
        if !(i in cameras)
            continue
        end
        frames_folder = "$(file[1:end-4])_frames"

        mkpath(frames_folder)

        smallfile = joinpath(folder, "small_$(i-1).avi")

        if force || !isfile(smallfile)
            @info "Downsizing $file into $smallfile"
            downsizevideo(file, smallfile)
            @info "Done."
        end

        if force || (length(readdir(frames_folder)) == 0)
            @info "Converting $smallfile to frames..."
            videotoframes(smallfile, frames_folder)
            @info "Done."
        end

        data[i] = Dict(
            :path => frames_folder,
            :files => sort(readdir(frames_folder)),
        )
    end

    return data
end


function load_annotations(file; cameras = 1:14)
    data = Dict()
    mat = matread(file)

    for i in cameras
        data[i] = Dict(
            :annot2d => mat["annot2"][i],
            :annot3d => mat["annot3"][i],
            :univannot3d => mat["univ_annot3"][i],
        )
    end
    return data
end


# Parsing

function parsecalibrationfile(file)
    lines = readlines(file)[2:end]
    Ks = SMatrix{3, 3}[]
    RTs = SMatrix{3, 4}[]
    for (c, i) in enumerate(1:7:length(lines)-1)
        intrinsic::String = lines[i+4][15:end]
        extrinsic::String = lines[i+5][15:end]
        K = SMatrix{3, 3}(reshape(parse.(Float32, split(intrinsic)), 4, 4)'[1:3, 1:3])
        RT = SMatrix{3, 4}(reshape(parse.(Float32, split(extrinsic)), 4, 4)'[1:3, 1:4])
        push!(Ks, K)
        push!(RTs, RT)
    end
    return Ks, RTs
end


function parsepose2d(a, factor = 4.)
    k = length(a) ÷ 2  # number of keypoints
    pose = Vector{SVector{2}}(undef, k)

    for (i, idx) in enumerate(1:2:2k)
        x, y = a[idx], a[idx+1]
        # switch (x, y) to (y, x) and add 1 because arrays start at 1 in Julia
        #pose[i] = SVector{2}(a[2i]+1, a[2i-1]+1) / factor
        pose[i] = SVector{2}(x, y)
    end
    return pose
end


function parsepose3d(a, factor = 4.)
    k = length(a) ÷ 3  # number of keypoints
    pose = Vector{SVector{3}}(undef, k)
    for (i, idx) in enumerate(1:3:3k)
        x, y, z = a[idx], a[idx+1], a[idx+2]
        pose[i] = SVector{3}(x, y, z)
        #pose[i] = SVector{3}(a[3i-2], a[3i], -a[3i-1]) / factor
    end
    return pose
end

# Utilities

videoordering(file) = parse(Int, splitpath(file)[end][7:end-4])


function videotoframes(file, dst)
    mkpath(dst)
    cmd = `ffmpeg -i "$file" -qscale:v 1 "$(joinpath(dst, "img_0_%06d.jpg"))"`
    run(cmd)
end


function downsizevideo(src, dst)
    cmd = `ffmpeg -y -i "$src" -vf scale=512:-2,setsar=1:1 -c:v libx264 -c:a copy "$dst"`
    run(cmd)
end




end # module
