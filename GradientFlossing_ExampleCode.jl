using BackwardsLinalg,LinearAlgebra,Flux,Random,RandomNumbers.Xorshifts,Statistics
using Flux.Data: DataLoader
using Zygote: @ignore, gradient
LinearAlgebra.BLAS.set_num_threads(4)
using BSON: @load, @save
import Flux.Losses.logitbinarycrossentropy#, Flux.Losses.mse
# This script trains a vanilla RNN on a XOR task
include("GradientFlossing_XOR.jl")
function trainRNNflossing(N, E, Ef, Ei, Ep, Ni, B, S, T, Tp, Ti, sIC, sIn, sNet, sONS, lr, b1, b2, IC, g, gbar, I1, delay, wsS, wsM, wrS, wrM, bS, bM, nLE, task, intype, Lwnt)
    flush(stdout)
    phi=tanh
    if task > 0
        firstlossidx = Int(2 * delay)
    else
        firstlossidx = Int(delay * 2^abs(task))
    end
    @assert T > firstlossidx "T has to be > firstlossidx. Make T bigger"
    @assert N >= nLE "N has to be > nLE"
    decay = 0#0.0001f0 # no weight decay for now
    lstarget = Float32(Lwnt) * ones(Float32, nLE)
    onlyLEdiagnostic = false
    LEdiagnostic = false
    LmaxAll = Float32[]
    LlastAll = Float32[]
    LspectrumAll = Float32[]
    nStepTransient = 10#firstlossidx
    nstepONS = 1
    nStep = Tp #number of timesteps
    nStepTransientONS = ceil(Int, nStep / 10) # steps during transient of ONS
    phid(x) = sech(x)^2
    function getD(x, J)
        phidtemp = phid.(x)
        D = Diagonal(vec(phidtemp)) * J
        return D
    end
    R = 1 # number of readouts
    onlyfinaltesterror = false
    accuracy_interval = 100
    println("T:", T)
    
    s, rtarg = generateInputOutput(B, T, sIn, S, I1, delay, task)#(B, T, sIn, S, sigma,tauS,dt)
    println("generating weights")
    Random.seed!(sNet)
    ws = (wsS * randn(Float32, N, S) + wsM * ones(Float32, N, S))
    wr = wrS * randn(Float32, R, N) + wrM * ones(Float32, R, N)
    b = bM .+ bS * rand(Float32, N)
    offset = [0.001f0]
    wsInit = copy(ws)
    wrInit = copy(wr)
    bInit = copy(b)
    offsetInit = copy(offset)
    offsetInit = copy(offset)
    if IC == 1 #Gaussian recurrent coupling
        J = Float32(g) * randn(Float32, N, N) / Float32(sqrt(N)) .+ Float32(gbar / N)
    elseif IC == 3 #Orthogonal recurrent coupling 
        J = Float32(g) * randn(Float32, N, N) / Float32(sqrt(N))
        qrout = qr(J)
        J = Float32(g) * qrout.Q * Matrix(I, size(J)...) #turns it into a matrix
    else
        println("WARNING: NO IC defined")
    end

    function calclossbinarycrossentropy(s, rtarg)
        loss = 0
        x = copy(xInit) #+ sig*randn(Float32, N, size(s,3))
        for ti = 1:T
            x = J * phi.(x) .+ b + ws * s[ti, :, :] / S
            if ti >= firstlossidx
                r = wr * x .+ offset
                loss += logitbinarycrossentropy(r, rtarg[ti, :, :]; agg = sum)#sum(abs2.(r - ))  #/NT#loss # loss is MSE
            end
        end
        return loss / B / (T - firstlossidx + 1) / size(wr, 1) #+ norm(J)/300 #+ norm(b)/1000  + norm(wr)/10000 # + norm(wr)/1000
    end

    function getAccuracy(s, rtarg)
        accuracy = 0
        x = copy(xInit) #+ sig*randn(Float32, N, size(s,3))
        for ti = 1:T
            x = J * phi.(x) .+ b + ws * s[ti, :, :] / S
            if ti >= firstlossidx
                r = wr * x .+ offset
                accuracy += sum((sigmoid.(r) .>= 0.5) .== rtarg[ti, :, :])#sum(abs2.(r - ))  #/NT#loss # loss is MSE
            end
        end
        return accuracy / B / (T - firstlossidx + 1) / size(wr, 1) #+ norm(J)/300 #+ norm(b)/1000  + norm(wr)/10000 # + norm(wr)/1000
    end

        function trainWithAccuracy!(loss, ps, data, opt)
        local training_loss
        psZygote = Flux.Params(ps)
        for d in data
            gs = gradient(psZygote) do
                training_loss = loss(d...)
                return training_loss
            end
            accuracyNow = getAccuracy(d...)
            Flux.update!(opt, psZygote, gs)
            svdJ = svdvals(gs[J])
            svdxInit = svdvals(gs[xInit])
        end
    end

    function trainLSappend!(psFlossing, data, opt)
        local training_loss
        local ls
        psZygote = Flux.Params(psFlossing)
        for d in data
            gs = gradient(psZygote) do
                ls = calcLyapunovSpectrum(d...)
                training_loss = mean(abs2.(ls .- lstarget))
                return training_loss
            end
            Flux.update!(opt, psZygote, gs)
        end
    end
            #########################################
            # THIS calculates the Lyapunov spectrum #
            #########################################
function calcLyapunovSpectrum(s, rtarg; xInit=xInit, T=Tp, nLE=nLE)
        x = copy(xInit)[:, 1]
        tsim = 0
        LS = zeros(Float32, nLE) # initialize Lyapunov spectrum
        Random.seed!(sONS)
        Qinit, Rinit = BackwardsLinalg.qr(randn(Float32, N, nLE)) # initialize orthonormal system
        Q = Matrix(Qinit)
        R = Matrix(Rinit)
        diagR = diagind(R)
        for n = 1:Tp
            x = J * phi.(x) .+ b + ws * s[n, :, :] / S
            D = getD(x, J)
            Q = D * Q
            if nLE > 0 && (n - 1) % nstepONS == 0
                Qout, Rout = BackwardsLinalg.qr(Q)
                Q = Matrix(Qout)
                R = Matrix(Rout)
                if n > nStepTransient + nStepTransientONS
                    logabsRhere = log.(abs.(R[diagR]))
                    LS += logabsRhere
                    tsim += nstepONS
                end
            end
        end
        if nLE > 0 && nStep % nstepONS != 0
            Q, R = BackwardsLinalg.qr(Q)
            LS += log.(abs.(R[diagR]))
        end
        Lspectrum = LS / tsim
        return Lspectrum
    end
    
    function relaxIC(sIC,steps)
        Random.seed!(sIC)
        x = randn(Float32, N, B)
        for i = 1:steps
            x = J * phi.(x) .+ b 
        end
        return x
    end    
    
    
    Random.seed!(sIC)
    xInit = Float32.(relaxIC(sIC, round(Int, 100)))
    opt = Flux.Optimiser(ClipNorm(0.03f0), ADAMW(lr, (b1, b2), decay))
    optFlossing = ADAMW(lr, (b1, b2), decay)
    ps = (ws, J, wr, b, offset, xInit)
    psFlossing = (ws, J, b)
    prevt = time()
    testError = Float32[]
    
    startWithPreflossing = Ep > 0 
    ni = 0 # counter for interflossing
    @time for ei = 1:E
        print(ei, "\r")
        if (ei == 1 && startWithPreflossing) || ((mod(ei, Ef) == 0) && (ni < Ni)) ### for periodic interflossing
            Bin = B
            B = 1
            if ei == 1
                NumberOfFlossingEpochs = Ep
                println("preflossing now")
            else
                NumberOfFlossingEpochs = Ei
                println("interflossing now at $ei")
                ni += 1
            end
            #############################
            # THIS IS THE FLOSSING LOOP #
            #############################

            @time for i = 1:NumberOfFlossingEpochs
                print(i, "\r")
                if nLE > 0
                    s, rtarg = generateInputOutput(B, Ti, E * sIn + ei + i, S, I1, delay, task)
                    trainLSappend!(psFlossing, DataLoader(s, rtarg; batchsize=B, shuffle=true), optFlossing)
                end
            end
                    B = Bin
        end
        ###################################
        # THIS IS THE USUAL TRAINING LOOP #
        ###################################
        Random.seed!(ei)
        s, rtarg = generateInputOutput(B, T, E * sIn + ei, S, I1, delay, task)
        trainWithAccuracy!(calclossbinarycrossentropy, ps, DataLoader(s, rtarg; batchsize=B, shuffle=true), opt)
    end
end
