%% DE INTREBAT:
%1. Ce se ploteaza: abs(fft) sau abs(fft)^2?

%2. Cand calculam cât la sută din coeficienti compun pragurile alea de
%energie, procentul e din ce, din toți coeficienții FFT, sau doar din cei
%ai primei jumătăți?

%3. Să scalăm FFT-ul cu Fs sau Nr_esantioane? sau deloc?

%4. Cum plotam procentele de energie in 2D?

%6. Cum sa plotez componentele TKL-ului? Valoriile proprii? Nu pot sa le
%ordonez pe toate pentru ca n-am destula memorie. Dar in 2D?

%7. La fel la Haar? Ce componente plotez?
%#ok<*AGROW>
%#ok<*INUSD>
classdef    functii
    methods     ( Static = true )
        
        function plot_gray(imagine_fft)
            [m,n] = size(imagine_fft);
            A = db(abs(imagine_fft));
            colormap jet
            imagesc(A)
            title("Spectrul imaginii b/w ")
            
            ox = (0:n-1) ./ n;
            oy = (0:m-1) ./ m;
            
            figure;
            title("Spectrul imaginii b/w (3D)")
            [X, Y] = meshgrid(ox, oy);
            colormap jet
            h = surf(X, Y, A, 'FaceColor','interp');
            set(h,'LineStyle','none')
            set(gca,'Xdir','reverse','Ydir','reverse')
        end
        
        function plot_audio(Fs, orig, z)
            T = 1/Fs;
            t = (0:length(orig)-1)*T;
            
            figure
            plot(t, z, 'r')
            hold on
            plot(t, orig, 'b')
            xlabel("t (secunde)")
            ylabel("Amplitudine")
            legend("Reconstruit", "Oriignal")
        end
        
        function plot_rgb(imagine_fft)
            [m,n,~] = size(imagine_fft);
            
            R = db(abs(imagine_fft(:, :, 1)));
            G = db(abs(imagine_fft(:, :, 2)));
            B = db(abs(imagine_fft(:, :, 3)));
            
            figure
            title("Spectrul imaginii rgb")
            colormap jet
            subplot(3, 1, 1)
            imagesc(R)
            subplot(3, 1, 2)
            imagesc(G)
            subplot(3, 1, 3)
            imagesc(B)
            
            ox = (0:n-1) ./ n;
            oy = (0:m-1) ./ m;
            
            figure
            title("Spectrul imaginii rgb (3D)")
            subplot(3, 1, 1)
            [X, Y] = meshgrid(ox, oy);
            colormap jet
            h = surf(Y, X, R, 'FaceColor','interp');
            set(h,'LineStyle','none')
            set(gca,'Xdir','reverse','Ydir','reverse')
            
            subplot(3, 1, 2)
            [X, Y] = meshgrid(ox, oy);
            colormap jet
            h = surf(X, Y, G, 'FaceColor','interp');
            set(h,'LineStyle','none')
            set(gca,'Xdir','reverse','Ydir','reverse')
            
            subplot(3, 1, 3)
            [X, Y] = meshgrid(ox, oy);
            colormap jet
            h = surf(X, Y, B, 'FaceColor','interp');
            set(h,'LineStyle','none')
            set(gca,'Xdir','reverse','Ydir','reverse')
        end
        
        function [f, P1] = plot_1d(audio_fft, Fs, oy)
            if strcmp(oy, 'abs')
                P2 = abs(audio_fft);
                P1 = P2(1:length(audio_fft)/2+1);
                P1(2:end-1) = 2*P1(2:end-1);
            elseif strcmp(oy, 'power')
                P2 = audio_fft(1:length(audio_fft)/2+1);
                P1 = abs(P2).^2;
                P1(2:end-1) = 2*P1(2:end-1);
                P1 = db(P1);
            end
            
            f = Fs/length(audio_fft) * (0:(length(audio_fft)/2));
            f = f / Fs;
            
            figure
            plot(f, P1);
            title("Complex Magnitude of fft Spectrum")
            xlabel("Frecventa normalizata")
            if strcmp(oy, 'abs')
                ylabel("|fft(X)|")
            elseif strcmp(oy, 'power')
                ylabel("Spectral Power [dB]")
            end
        end
        
        function plot_1d_segmente(bucatiTransf, Transf)
            %ELIMINARE PARTE DE REST (ULTIM SEGMENT)
            bucatiTransf = bucatiTransf(1 : end-1);
            
            ox = (0:size(bucatiTransf, 2)-1);
            oy = (0:size(bucatiTransf{1},1)-1);
            
            bucatiTransf = cell2mat(bucatiTransf);
            %     bucatiTransf = abs(bucatiTransf);
            
            [X, Y] = meshgrid(oy, ox);
            colormap jet
            h = surf(X, Y, bucatiTransf', 'FaceColor','interp');
            colorbar
            title("Magnitude of " + Transf + " Spectrum")
            
            if(strcmp(Transf,'Haar2D') == 1)
                xlabel('Index Linie')
                ylabel('Index Coloana')
            else
                xlabel("Index coeficienti " + Transf)
                ylabel("Index Segment")
            end
            zlabel("Magnitudine coeficient " + Transf)
            set(h,'LineStyle','none')
            set(gca,'Xdir','reverse','Ydir','reverse')
        end
        
        
        function norma_eroare = plot_eroare_1d(orig, rec, Fs)
            er = abs(orig - rec);
            T = 1 / Fs;
            
            norma_eroare = norm(orig-rec);
            
            figure
            t = (0:length(orig)-1)*T;
            plot(t,er)
            title("Eroarea dintre semnalul original si cel reconstruit")
            xlabel("t (seconds)")
            ylabel("y(t)")
            gtext(['Norma erorii: ' num2str(norma_eroare)]);
            
        end
        
        
        function [energie,coefV, procente_coef, indici] = proc_energie_2d(coef)
            % coefV = [];
            
            if size(coef, 2) == 1
                coefV{1} = reshape(coef{1},[],1);
            else
                for i = 1 : size(coef, 2)
                    coefV{i} = reshape(coef{i}, [], 1);
                end
            end
            
            for i = 1 : size(coefV, 2)
                energie{i} = abs(coefV{i}).^2;
                energie{i} = sort(energie{i}, 'descend');
                energie_totala{i} = sum(energie{i});
            end
            
            procente = [0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.99];
            indici = {};
            
            for i = 1 : size(energie, 2)
                indiceEnergie = 0;
                suma = 0;
                for j = 1 : length(procente)
                    enteredLoop = false;
                    while (suma < procente(j) * energie_totala{i}) & indiceEnergie < length(energie{i})
                        indiceEnergie = indiceEnergie + 1;
                        indici{i}(j) = indiceEnergie;
                        suma = suma + energie{i}(indiceEnergie);
                        enteredLoop = true;
                    end
                    if j > 1 && enteredLoop == false
                        indici{i}(j) = indici{i}(j-1);
                    end
                end
            end
            
            procente_coef = {};
            for i = 1 : size(energie, 2)
                figure
                plot(energie{i})
                for j = 1 : length(indici{i})
                    procente_coef{i}(j) = indici{i}(j)/length(coefV{i});
                    xline(indici{i}(j), 'r--', [num2str(procente(j)*100) '% - Procent coef.: ' num2str(procente_coef{i}(j))], 'LineWidth', 1);
                end
            end
        end
        
        function plot_1d_segmente_hwt(bucatiTransf, Transf, lungSegment)
            numSegments = length(bucatiTransf) / lungSegment;
            
            % Reshape vectorul într-o matrice 2D
            bucatiTransf = reshape(bucatiTransf, [lungSegment, numSegments]);
            
            % Convertim la valorile absolute pentru vizualizare
            bucatiTransf = abs(bucatiTransf);
            % Crearea figurii
            figure
            colormap jet
            ox = (0:numSegments-1); % Indexul segmentului
            oy = (0:lungSegment-1); % Indexul coeficientului în segment
            
            [X, Y] = meshgrid(oy, ox);
            h = surf(X, Y, bucatiTransf', 'FaceColor','interp');
            colorbar
            title("Magnitude of " + Transf + " Spectrum")
            % xlim([0 lungSegment-1])
            % ylim([0 numSegments-1])
            % Setarea etichetelor axelor
            xlabel("Index coeficient " + Transf)
            ylabel("Index Segment")
            zlabel("Magnitudine coeficient " + Transf)
            
            % Îndepărtarea liniilor de pe suprafață pentru claritate
            set(h,'LineStyle','none')
            
            % Inversarea direcției axelor pentru a se potrivi cu convenția obișnuită
            set(gca,'Xdir','reverse','Ydir','reverse')
        end
        
        
        
        
        function plot_eroare_2d(orig, rec)
            % Define the axes based on the original image dimensions
            oy = 0:size(orig, 1)-1; % y-axis corresponds to the number of rows
            ox = 0:size(orig, 2)-1; % x-axis corresponds to the number of columns
            [X, Y] = meshgrid(ox, oy); % Create a grid that matches the image dimensions
            
            % Calculate the absolute difference error
            er = imabsdiff(orig, rec);
            
            
            % Check if the image is grayscale or color
            if size(orig, 3) == 1
                % It's a grayscale image
                figure;
                colormap jet;
                h = surf(X, Y, double(er), 'FaceColor', 'interp'); % Make sure 'er' is double
                set(h, 'LineStyle', 'none');
                xlabel('Index Coloana');
                ylabel('Index Linie');
                zlabel('Eroare');
                set(gca, 'Xdir', 'reverse', 'Ydir', 'reverse');
                title('Eroarea 2D pentru imaginea monocroma');
                colorbar;
                % Adjust the position of the text
                textPositionX = max(get(gca, 'XLim')) * 0.8;
                textPositionY = max(get(gca, 'YLim')) * 0.1;
                normaErorii = sum(er(:));
                text(textPositionX, textPositionY, max(er(:)), ['Norma erorii: ' num2str(normaErorii)]);
            else
                % It's a color image, plot errors for each channel
                er1 = imabsdiff(orig(:, :, 1), rec(:, :, 1));
                er2 = imabsdiff(orig(:, :, 2), rec(:, :, 2));
                er3 = imabsdiff(orig(:, :, 3), rec(:, :, 3));
                
                figure;
                colormap jet;
                
                % Plot error for the Red channel
                subplot(3, 1, 1);
                h = surf(X, Y, double(er1), 'FaceColor', 'interp');
                set(h, 'LineStyle', 'none');
                xlabel('Index Coloana');
                ylabel('Index Linie');
                zlabel('Eroare Rosu');
                title('Eroarea 2D pentru canalul Rosu');
                colorbar;
                
                % Plot error for the Green channel
                subplot(3, 1, 2);
                h = surf(X, Y, double(er2), 'FaceColor', 'interp');
                set(h, 'LineStyle', 'none');
                xlabel('Index Coloana');
                ylabel('Index Linie');
                zlabel('Eroare Verde');
                title('Eroarea 2D pentru canalul Verde');
                colorbar;
                
                % Plot error for the Blue channel
                subplot(3, 1, 3);
                h = surf(X, Y, double(er3), 'FaceColor', 'interp');
                set(h, 'LineStyle', 'none');
                xlabel('Index Coloana');
                ylabel('Index Linie');
                zlabel('Eroare Albastru');
                title('Eroarea 2D pentru canalul Albastru');
                colorbar;
            end
        end
        
        function [fftizat_jum, energie, procente_coef] = proc_energie_fft(fftizat, Fs)
            fftizat_jum = fftizat;
            fftizat_jum = fftizat_jum(1:length(fftizat)/2+1);
            fftizat_jum(2:end-1) = 2*fftizat_jum(2:end-1);
            
            energie = abs(fftizat_jum).^2;
            energie_totala = sum(energie);
            
            procente = [0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.99];
            indici = zeros(12, 1);
            
            suma = 0;
            j = 0;
            
            [f, ~] = functii.plot_1d(fftizat, Fs, 'power');
            
            for i = 1 : length(procente)
                while (suma <= procente(i) * energie_totala) & j < length(fftizat_jum)
                    j = j + 1;
                    suma = suma + energie(j);
                end
                indici(i) = j-1;
            end
            
            procente_coef = zeros(length(indici), 1);
            
            for i = 1 : length(indici)
                procente_coef(i) = indici(i)/length(fftizat_jum);
                xline(f(indici(i)), 'r--', [num2str(procente(i)*100) '% - Procent coef.: ' num2str(procente_coef(i))], 'LineWidth', 1);
            end
        end
        
        function [cumulative_energy, procente_coef] = proc_energie_klt(D)
            proportion_variance = D / sum(D);
            cumulative_energy = cumsum(proportion_variance);
            procente = [0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.99];
            procente_coef = arrayfun(@(x) find(cumulative_energy >= x, 1, 'first'), procente);
            
            figure;
            plot(cumulative_energy);
            xlabel('Number of Components');
            ylabel('Cumulative Variance Explained');
            title('Scree Plot');
            grid on;
            hold on;
            for i = 1:length(procente)
                xline(procente_coef(i), 'r--', [' ' num2str(procente(i)*100) '%'], 'LineWidth', 1);
            end
            hold off;
        end
        
        function [bucati, rest] = segmentare_bucati(audio, len_bucati)
            bucati = [];
            rest = [];
            
            if size(audio, 1) > len_bucati
                for i = 1 : size(audio, 1) / len_bucati
                    bucati{i} = audio((i-1)*len_bucati+1:i*len_bucati);
                end
                rest = audio(i*len_bucati+1:end);
            end
        end
        
        function [audio, audio_fft, Fs] = fft1d(x)
            [audio, Fs] = audioread(x);
            audio_fft = fft(audio);
            
            T = 1 / Fs;
            
            figure
            t = (0:length(audio)-1)*T;
            plot(t,audio)
            title("Semnalul audio original")
            xlabel("t (seconds)")
            ylabel("y(t)")
            
            functii.plot_1d(audio_fft, Fs, 'abs');
        end
        
        function y = inv_fft1d(x, Fs)
            y = ifft(x);
            T = 1 / Fs;
            
            figure
            t = (0:length(y)-1)*T;
            plot(t,y)
            title("Semnalul audio reconstruit")
            xlabel("t (seconds)")
            ylabel("y(t)")
        end
        
        function [imagine, imagine_fft] = fft2d(image)
            imagine = imread(image);
            [~, ~, ch] = size(imagine);
            
            imagine_fft = fft2(double(imagine));
            
            if(ch == 3)
                functii.plot_rgb(imagine_fft);
            else
                functii.plot_gray(imagine_fft);
            end
        end
        
        function y = inv_fft2d(x)
            y = uint8(ifft2(x));
            y = real(y);
            
            figure
            title("Imaginea reconstruita")
            imagesc(abs(y))
            
            if(size(y, 3) ~= 3)
                colormap('gray')
            end
        end
        
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
            [bucati, rest] = functii.segmentare_bucati(audio, 1000);
            
            if(~isempty(bucati))
                for i = 1 : size(bucati, 2)
                    [y{i}, D{i}, Vm{i}, xM{i}] = functii.proc_tkl1d(bucati{i});
                end
                if(~isempty(rest))
                    [y{i+1}, D{i+1}, Vm{i+1}, xM{i+1}] = functii.proc_tkl1d(rest);
                end
            else
                [y, D, Vm, xM] = functii.proc_tkl1d(audio);
            end
            
            functii.plot_1d_segmente(y, "TKL")
            
        end
        
        function [imagine, y, Vm, xM, xdim, ydim] = tkl2d(image)
            imagine = imread(image);
            imagine = double(imagine);
            [xdim, ydim, ~] = size(imagine);
            imgVec = reshape(imagine, 1, []);
            [bucati, rest] = functii.segmentare_bucati(imgVec', 1000);
            
            if(~isempty(bucati))
                for i = 1 : size(bucati, 2)
                    [y{i}, D{i}, Vm{i}, xM{i}] = functii.proc_tkl1d(bucati{i});
                end
                if(~isempty(rest))
                    [y{i+1}, D{i+1}, Vm{i+1}, xM{i+1}] = functii.proc_tkl1d(rest);
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
            imgVec = functii.inv_tkl1d(x, Vm, xM);
            y = reshape(imgVec, xdim, ydim);
        end
        
        function [y, huri, r] = proc_haar1d(x)
            L = floor(log2(size(x, 1)));
            
            h = [1 1];
            g = [-1 1];
            
            for i = 1 : L
                huri{i} = conv(x, g);
                huri{i} = huri{i}(2:2:end);
                
                r = conv(x, h);
                r = r(2:2:end);
                x = r;
                
                if(size(huri{i}, 1) < size(r, 1))
                    huri{i} = [huri{i}; 0];
                elseif(size(huri{i}, 1) > size(r, 1))
                    r = [r; 0];
                end
            end
            
            huriT = cell2mat(huri');
            huriT = flip(huriT);
            y = [r; huriT];
            
        end
        
        function [audio, y, huri, r, Fs] = haar1d(x, lungSegment)
            [audio, Fs] = audioread(x);
            
            [bucati, rest] = functii.segmentare_bucati(audio, lungSegment);
            
            
            if(~isempty(bucati))
                for i = 1 : size(bucati, 2)
                    [y{i}, huri{i}, r{i}] = functii.proc_haar1d(bucati{i});
                end
                if(~isempty(rest))
                    [y{i+1}, huri{i+1}, r{i+1}] = functii.proc_haar1d(rest);
                end
            end
            
            figure
            functii.plot_1d_segmente(y, "Haar")
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
                    
                    r_prev = conv(functii.interpolare(rCurent), h) + conv(functii.interpolare(huri{i}{j}), g);
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
                [coef_col{col}, huri_col{col}, r_col{col}] = functii.proc_haar1d(colCurenta);
            end
            coef_col_temp = cell2mat(coef_col);
            
            for row = 1 : size(coef_col_temp, 1)
                rowCurent = coef_col_temp(row, :)';
                [coef{row}, huri{row}, r{row}] = functii.proc_haar1d(rowCurent);
            end
        end
        
        function [imagine, coef, huri, r, huri_col, r_col] = haar2d(image)
            imagine = imread(image);
            imagine = double(imagine);
            [~, ~, ch] = size(imagine);
            
            for i = 1 : ch
                [coef{i}, huri{i}, r{i}, coef_col{i}, huri_col{i}, r_col{i}] = functii.proc_haar2d(imagine(:, :, i));
            end
            
            figure
            for i = 1 : ch
                subplot(2, ch, i)
                functii.plot_1d_segmente(coef_col{i}, "Haar2D")
                subplot(2, ch, i+ch)
                functii.plot_1d_segmente(coef{i}, "Haar2D")
                coef{i} = cell2mat(coef{i});
            end
            
        end
        
        function [x] = proc_inv_haar2d(huri, r, huri_col, r_col)
            h = [1 1] / 2;
            g = [1 -1] / 2;
            
            L = size(huri{1},2);
            
            for i = 1 :size(r, 2)
                rCurent = r{i};
                for j = L : -1 : 1
                    if(size(huri{i}{j}, 1) < size(rCurent, 1))
                        rCurent = rCurent(1:end-1);
                    end
                    r_prev = conv(functii.interpolare(rCurent), h) + conv(functii.interpolare(huri{i}{j}), g);
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
                    r_prev = conv(functii.interpolare(rCurent), h) + conv(functii.interpolare(huri_col{i}{j}), g);
                    rCurent = r_prev;
                end
                x{i} = rCurent;
            end
            
            x = cell2mat(x);
        end
        
        function [x] = inv_haar2d(huri, r, huri_col, r_col)
            for i = 1 : size(huri, 2)
                [x{i}] = functii.proc_inv_haar2d(huri{i}, r{i}, huri_col{i}, r_col{i});
            end
            
            if size(huri, 2) ~= 1
                x = cat(3, x{1}, x{2}, x{3});
            else
                x = x{1};
            end
        end
        
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
            H = functii.hadamardMatrix(lungSegment);
            
            walshMatrix = permMatrix * H;
            
            [bucati, rest] = functii.segmentare_bucati(audio, lungSegment);
            
            if(~isempty(bucati))
                for i = 1 : size(bucati, 2)
                    y{i} = functii.proc_wht1d(bucati{i}, walshMatrix);
                end
                if(~isempty(rest))
                    %%Padding cu 0-uri
                    rest = [rest; zeros(lungSegment - length(rest), 1)];
                    y{i+1} = functii.proc_wht1d(rest, walshMatrix);
                end
            end
        end
        
        function H = hadamardMatrix(order)
            if order == 0
                error('Ordinul matricei Hadamard nu poate fi 0');
            end
            if order == 1
                H = 1;
            else
                H_half = functii.hadamardMatrix(order / 2);
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
            H = functii.hadamardMatrix(lungSegment_col);
            walshMatrix_col = permMatrix_col * H;
            
            if ch == 1
                % Apply 1D WHT on each column
                for col = 1 : ydim_padded
                    colCurenta = imagine(:, col);
                    y_col{col} = walshMatrix_col * colCurenta;
                end
                y_col_mat = cell2mat(y_col);
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
            H = functii.hadamardMatrix(lungSegment_row);
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
            
            % figure
            for i = 1 : ch
                % subplot(2, ch, i)
                % functii.plot_1d_segmente(y_col{i}, "Haar2D")
                % subplot(2, ch, i+ch)
                % functii.plot_1d_segmente(y{i}, "Haar2D")
                y{i} = cell2mat(y{i});
            end
        end
        
        function y = inv_wht2d(x, walshMatrix_col, walshMatrix_row, xdim, ydim, xdim_orig, ydim_orig)
            if size(x, 2) == 3
                x_mat = cat(3, x{1}, x{2}, x{3});
            elseif size(x, 2) == 1
                x_mat = x{1};
            end
            size(x_mat)
            
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