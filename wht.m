classdef    wht
    methods     ( Static = true )
        function y = proc_wht1d(x, walshMatrix)
            y = walshMatrix * x;
        end
        
        function [audio, y, Fs, walshMatrix] = wht1d(x, lungSegment)
            if log2(lungSegment) ~= floor(log2(lungSegment))
                error('Lungimea segmentelor nu este o putere a lui 2');
            end
            
            [audio, Fs] = audioread(x);
            
            order = log2(lungSegment);
            permMatrix = hada2walsh_matrix(order);
            H = wht.hadamardMatrix(lungSegment);
            
            walshMatrix = permMatrix * H;
            
            [bucati, rest] = utils.segmentare_bucati(audio, lungSegment);
            
            if(~isempty(bucati))
                for i = 1 : size(bucati, 2)
                    y{i} = wht.proc_wht1d(bucati{i}, walshMatrix);
                end
                if(~isempty(rest))
                    %%Padding cu 0-uri
                    rest = [rest; zeros(lungSegment - length(rest), 1)];
                    y{i+1} = wht.proc_wht1d(rest, walshMatrix);
                end
            end

            figure
            utils.plot_1d_segmente(y, "WHT1D")
        end
        
        function H = hadamardMatrix(order)
            if order == 0
                error('Ordinul matricei Hadamard nu poate fi 0');
            end
            if order == 1
                H = 1;
            else
                H_half = wht.hadamardMatrix(order / 2);
                H = [H_half, H_half; H_half, -H_half];
            end
        end
        
        function y = inv_wht1d(x, walshMatrix)
            N = size(walshMatrix, 1);
            inverseMatrix = (1/N) * walshMatrix';
            
            for i = 1:size(x, 2)
                y_segment_inv = inverseMatrix * x{i};
                y_inv{i} = y_segment_inv;
            end
            
            y = cat(1, y_inv{:});
        end
        
        function [y, orig, walshMatrix_col, walshMatrix_row, xdim_padded, ydim_padded, xdim, ydim] = wht2d(image)
            orig = imread(image);
            %make the image only one channel
            [xdim, ydim, ch] = size(orig);
            % If image is RGB, skip for now
            
            imagine = double(orig);
            
            
            if ch == 1
                % Check if column and row dimensions are powers of 2, if not, pad with zeros to the next power of 2
                if log2(xdim) ~= floor(log2(xdim))
                    imagine = [imagine; zeros(2^(ceil(log2(xdim))) - xdim, ydim)];
                end
                xdim_padded = size(imagine, 1);
                
                if log2(ydim) ~= floor(log2(ydim))
                    imagine = [imagine, zeros(xdim_padded, 2^(ceil(log2(ydim))) - ydim)];
                end
                ydim_padded = size(imagine, 2);
            elseif ch == 3
                if log2(xdim) ~= floor(log2(xdim))
                    imagine = [imagine; zeros(2^(ceil(log2(xdim))) - xdim, ydim, 3)];
                end
                xdim_padded = size(imagine, 1);
                
                if log2(ydim) ~= floor(log2(ydim))
                    imagine = [imagine, zeros(xdim_padded, 2^(ceil(log2(ydim))) - ydim, 3)];
                end
                ydim_padded = size(imagine, 2);
            end
            
            lungSegment_col = xdim_padded;
            order_col = log2(lungSegment_col);
            permMatrix_col = hada2walsh_matrix(order_col);
            H = wht.hadamardMatrix(lungSegment_col);
            walshMatrix_col = permMatrix_col * H;
            
            if ch == 1
                % Apply 1D WHT on each column
                for col = 1 : ydim_padded
                    colCurenta = imagine(:, col);
                    y_col{1}{col} = walshMatrix_col * colCurenta;
                end
                y_col_mat = cell2mat(y_col{1});
            elseif ch == 3
                for c = 1 : ch
                    for col = 1 : ydim_padded
                        colCurenta = imagine(:, col, c);
                        y_col{c}{col} = walshMatrix_col * colCurenta;
                    end
                end
                y_col_r = cell2mat(y_col{1});
                y_col_g = cell2mat(y_col{2});
                y_col_b = cell2mat(y_col{3});
                y_col_cell = {y_col_r, y_col_g, y_col_b};
                y_col_mat = cat(3, y_col_r, y_col_g, y_col_b);
            end
            
            
            
            lungSegment_row = ydim_padded;
            order_row = log2(lungSegment_row);
            permMatrix_row = hada2walsh_matrix(order_row);
            H = wht.hadamardMatrix(lungSegment_row);
            walshMatrix_row = permMatrix_row * H;
            
            if ch == 1
                for row = 1 : size(y_col_mat, 1)
                    rowCurent = y_col_mat(row, :)';
                    y{1}{row} = walshMatrix_row * rowCurent;
                end
            elseif ch == 3
                for c = 1 : ch
                    y_col_c = y_col_mat(:, :, c);
                    for row = 1 : size(y_col_c, 1)
                        rowCurent = y_col_c(row, :)';
                        y{c}{row} = walshMatrix_row * rowCurent;
                    end
                end
            end
            size(y{1})
            
            figure
            for i = 1 : ch
                 subplot(2, ch, i)
                 utils.plot_1d_segmente(y_col{i}, "WHT2D")
                 subplot(2, ch, i+ch)
                 utils.plot_1d_segmente(y{i}, "WHT2D")
                 y{i} = cell2mat(y{i});
            end
        end
        
        function y = inv_wht2d(x, walshMatrix_col, walshMatrix_row, xdim, ydim, xdim_orig, ydim_orig)
            if size(x, 2) == 3
                x_mat = cat(3, x{1}, x{2}, x{3});
            elseif size(x, 2) == 1
                x_mat = x{1};
            end
            
            inverseMatrix_col = (1 / size(walshMatrix_col, 1)) * walshMatrix_col';
            inverseMatrix_row = (1 / size(walshMatrix_row, 1)) * walshMatrix_row';
            
            if size(x_mat, 3) == 1
                x_mat = x_mat';
                for i = 1:size(x_mat, 2)
                    x_mat(:, i) = inverseMatrix_col * x_mat(:, i);
                end
                
                x_mat = x_mat';
                inverseMatrix_row = (1 / size(walshMatrix_row, 1)) * walshMatrix_row';
                for i = 1:size(x_mat, 2)  % Operating on columns
                    x_mat(:, i) = inverseMatrix_row * x_mat(:, i);
                end
                y = uint8(x_mat)';
                y = y(1:xdim_orig, 1:ydim_orig);
            elseif size(x_mat, 3) == 3
                for c = 1 : 3
                    x_mat_c = x_mat(:, :, c)';
                    for i = 1:size(x_mat_c, 2)
                        x_mat_c(:, i) = inverseMatrix_col * x_mat_c(:, i);
                    end
                    
                    x_mat_c = x_mat_c';
                    
                    for i = 1:size(x_mat_c, 2)  % Operating on columns
                        x_mat_c(:, i) = inverseMatrix_row * x_mat_c(:, i);
                    end
                    y_c{c} = uint8(x_mat_c)';
                    y_c{c} = y_c{c}(1:xdim_orig, 1:ydim_orig);
                end
                y = cat(3, y_c{1}, y_c{2}, y_c{3});
            end
        end
    end
end