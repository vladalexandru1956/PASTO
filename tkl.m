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
            
            figure
            utils.plot_1d_segmente(y, "TKL")
            
        end
        
        function [imagine, y, Vm, xM, xdim, ydim] = tkl2d(image)
            imagine = imread(image);
            imagine = double(imagine);
            [xdim, ydim, ~] = size(imagine);
            imgVec = reshape(imagine, 1, []);
            [bucati, rest] = utils.segmentare_bucati(imgVec', 1000);
            
            if(~isempty(bucati))
                for i = 1 : size(bucati, 2)
                    [y{i}, D{i}, Vm{i}, xM{i}] = tkl.proc_tkl1d(bucati{i});
                end
                if(~isempty(rest))
                    [y{i+1}, D{i+1}, Vm{i+1}, xM{i+1}] = tkl.proc_tkl1d(rest);
                end
            end
        end
        
        function y = inv_tkl1d(x, Vm, xM)
            for i = 1 :size(x, 2)
                y{i} = Vm{i} * x{i} + xM{i};
            end
            
            y = cat(1, y{:});
        end
        
        function y = inv_tkl2d(x, Vm, xM, xdim, ydim)
            imgVec = tkl.inv_tkl1d(x, Vm, xM);
            y = reshape(imgVec, xdim, ydim);
        end
    end
end