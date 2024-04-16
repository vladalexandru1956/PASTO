classdef    tkl
    methods     ( Static = true )
        function [y, D, Vm, xM] = proc_tkl1d(x)
            xM = mean(x);
            x_centr = x - xM;
            [p_x, ~] = xcorr(x_centr, 'biased');
            
            R = fftshift(p_x); % in mod interesant, norma intre original si reconstruit da mai mica
                               % daca fac fftshift
            R = toeplitz(R(1 : (size(p_x) + 1) / 2));
            [V, D] = eig(R);
            
            [D, indx] = sort(diag(D), "descend");
            V = V(:, indx);
            
            Vm = V;
            y = Vm'*x_centr;
        end
        
        
        function [audio, y, D, Vm, xM, Fs] = tkl1d(x)
            [audio, Fs] = audioread(x);
            [bucati, rest] = utils.segmentare_bucati(audio, 1000);
            
            if(~isempty(bucati))
                for i = 1 : size(bucati, 2)
                    [y{i}, D{i}, Vm{i}, xM{i}] = tkl.proc_tkl1d(bucati{i});
                end
                if(~isempty(rest))
                    [y{i+1}, D{i+1}, Vm{i+1}, xM{i+1}] = tkl.proc_tkl1d(rest);
                end
            else
                [y, D, Vm, xM] = tkl.proc_tkl1d(audio);
            end

            T = 1 / Fs;
            
            figure
            t = (0:length(audio)-1)*T;
            plot(t,audio)
            title("Semnalul audio original")
            xlabel("t (seconds)")
            ylabel("y(t)")
            
            figure
            utils.plot_1d_segmente(y, "TKL")
            
        end
        
        function [imagine, y, Vm, xM, xdim, ydim] = tkl2d(image)
            imagine = imread(image);
            imagine = double(imagine);

            [xdim, ydim, ch] = size(imagine);
            if ch > 1
                canale = ['R', 'G', 'B'];
            else
                canale = [""];
            end

            for i = 1 : ch
                imgVec{i} = reshape(imagine(:, :, i), 1, []);
                [bucati{i}, rest{i}] = utils.segmentare_bucati(imgVec{i}', 1000);

                if(~isempty(bucati{i}))
                    for j = 1 : size(bucati{i}, 2)
                        [y{i}{j}, D{i}{j}, Vm{i}{j}, xM{i}{j}] = tkl.proc_tkl1d(bucati{i}{j});
                    end
                    if(~isempty(rest{i}))
                        [y{i}{j+1}, D{i}{j+1}, Vm{i}{j+1}, xM{i}{j+1}] = tkl.proc_tkl1d(rest{i});
                    end
                end

                if ch == 1
                    y{1} = y{i};
                end
            end

            figure
            for i = 1 : ch
                subplot(ch, 1, i)
                utils.plot_1d_segmente(y{i}, "TKL", canale(i))
            end
        end
        
        function y = inv_tkl1d(x, Vm, xM)
            for i = 1 :size(x, 2)
                y{i} = Vm{i} * x{i} + xM{i};
            end
            y = cat(1, y{:});
        end
        
        function y = inv_tkl2d(x, Vm, xM, xdim, ydim)
            for i = 1 : size(Vm, 2)
                imgVec{i} = tkl.inv_tkl1d(x{i}, Vm{i}, xM{i});
                y{i} = reshape(imgVec{i}, xdim, ydim);
            end

            if size(Vm, 2) ~= 1
                y = cat(3, y{1}, y{2}, y{3});
            else
                y = y{1};
            end
        end
    end
end