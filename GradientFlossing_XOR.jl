# this implements the copy task and XOR task in Julia

recabs(x)=length(x)==1 ? x[1] : abs(recabs(x[1:end-1])-x[end])

function generateInputOutput(B, T, seedInput, S, sigma, delay,task)
    target = zeros(Float32, T + delay, 1, B) # target output
    rng = Xoroshiro128Star(1)
    Random.seed!(rng, seedInput)
    s = bitrand(rng,T + delay, S, B)
    for bi = 1:B, si = 1:S, i = 1:(T+delay)
        if task==0 && i > (delay) # copy task:
            target[i, si, bi] =s[i-delay, si, bi]
        else  # hierarchical temporal XOR:
            h = abs(task)
            l = 2^h
            if i > (delay)*2^h
                target[i, si, bi] = recabs(reverse(s[i-delay:-delay:i-delay*2^h,si,bi]))
            end
        end
    end
    return s[1+delay:end, :, :]*Float32(sigma), target[1+delay:end, :, :]
end
