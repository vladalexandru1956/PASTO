classdef    haar
    methods     ( Static = true )
        function [y, huri, r] = proc_haar1d(x)
            L = log2(size(x, 1));
            if(rem(L,1) == 0)
                L = L - 1;
            else
                L = floor(L);
            end
            
            h = [1 1];
            g = [-1 1];
            
            for i = 1 : L
                huri{i} = conv(x, g);
                huri{i} = huri{i}(2:2:end);
                
                r = conv(x, h);
                r = r(2:2:end);
                x = r;
            end
            
            huriT = flip(huri');
            huriT = cell2mat(huriT);
            y = [r; huriT];
            
        end
        
        function [audio, y, huri, r, Fs] = haar1d(x, lungSegment)
            [audio, Fs] = audioread(x);
            
            [bucati, rest] = utils.segmentare_bucati(audio, lungSegment);
            
            
            if(~isempty(bucati))
                for i = 1 : size(bucati, 2)
                    [y{i}, huri{i}, r{i}] = haar.proc_haar1d(bucati{i});
                end
                if(~isempty(rest))
                    [y{i+1}, huri{i+1}, r{i+1}] = haar.proc_haar1d(rest);
                end
            end
            
            figure
            utils.plot_1d_segmente(y, "Haar")
        end
        
        function y = interpolare(x)
            y = upsample(x, 2);
            y = y(1:size(y) - 1);
        end
        
        function x_intarziat = inv_haar1d(huri, r)
            h = [1 1] / 2;
            g = [1 -1] / 2;
            
            for i = 1 :size(r, 2)
                rCurent = r{i};
                for j = size(huri{i}, 2) : -1 : 1
                    
                    if(size(huri{i}{j}, 1) < size(rCurent, 1))
                        rCurent = rCurent(1:end-1);
                    end
                    
                    r_prev = conv(haar.interpolare(rCurent), h) + conv(haar.interpolare(huri{i}{j}), g);
                    rCurent = r_prev;
                    
                end
                x_intarziat{i} = rCurent;
            end
            
            x_intarziat = cat(1, x_intarziat{:});
        end
        
        function [coef, huri, r, coef_col, huri_col, r_col] = proc_haar2d(imagine)
            [~, ydim, ~] = size(imagine);
            for col = 1 : ydim
                colCurenta = imagine(:, col);
                [coef_col{col}, huri_col{col}, r_col{col}] = haar.proc_haar1d(colCurenta);
            end
            coef_col_temp = cell2mat(coef_col);
            
            for row = 1 : size(coef_col_temp, 1)
                rowCurent = coef_col_temp(row, :)';
                [coef{row}, huri{row}, r{row}] = haar.proc_haar1d(rowCurent);
            end
        end
        
        function [imagine, coef, huri, r, huri_col, r_col] = haar2d(image)
            imagine = imread(image);
            imagine = double(imagine);
            [~, ~, ch] = size(imagine);
            
            for i = 1 : ch
                [coef{i}, huri{i}, r{i}, coef_col{i}, huri_col{i}, r_col{i}] = haar.proc_haar2d(imagine(:, :, i));
            end
            
            
            figure
            for i = 1 : ch
                subplot(2, ch, i)
                utils.plot_1d_segmente(coef_col{i}, "Haar2D")
                subplot(2, ch, i+ch)
                utils.plot_1d_segmente(coef{i}, "Haar2D")
                coef{i} = cell2mat(coef{i});
            end
        end
        
        function [x] = proc_inv_haar2d(huri, r, huri_col)
            h = [1 1] / 2;
            g = [1 -1] / 2;
            
            L = size(huri{1},2);
            
            for i = 1 :size(r, 2)
                rCurent = r{i};
                for j = L : -1 : 1
                    if(size(huri{i}{j}, 1) < size(rCurent, 1))
                        rCurent = rCurent(1:end-1);
                    end
                    r_prev = conv(haar.interpolare(rCurent), h) + conv(haar.interpolare(huri{i}{j}), g);
                    rCurent = r_prev;
                end
                x_inter{i} = rCurent;
            end
            
            X_inter = cell2mat(x_inter)';
            
            L = size(huri_col{1}, 2);
            
            for i = 1 : size(X_inter, 2)
                rCurent = X_inter(1:2, i);
                for j = L : -1 : 1
                    if(size(huri_col{i}{j}, 1) < size(rCurent, 1))
                        rCurent = rCurent(1:end-1);
                    end
                    r_prev = conv(haar.interpolare(rCurent), h) + conv(haar.interpolare(huri_col{i}{j}), g);
                    rCurent = r_prev;
                end
                x{i} = rCurent;
            end
            
            x = cell2mat(x);
        end
        
         function [x] = inv_haar2d(huri, r, huri_col)
            for i = 1 : size(huri, 2)
                [x{i}] = haar.proc_inv_haar2d(huri{i}, r{i}, huri_col{i});
            end
            
            if size(huri, 2) ~= 1
                x = cat(3, x{1}, x{2}, x{3});
            else
                x = x{1};
            end
        end
    end
end