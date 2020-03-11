
function groupby(xs, field)
    d = Dict()
    for x in xs
        f = getindex(x, field)
        if !haskey(d, f)
            d[f] = []
        end
        push!(d[f], x)
    end
    return d
end
