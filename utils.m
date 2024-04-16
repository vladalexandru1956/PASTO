%#ok<*AGROW>
%#ok<*INUSD>
classdef    utils
    methods     ( Static = true )
        
        function plot_gray(imagine_fft)
            [m,n] = size(imagine_fft);
            A = db(abs(imagine_fft));
            figure
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
            colormap jet
            subplot(3, 1, 1)
            imagesc(R)
            title("Spectrul canalului R")
            subplot(3, 1, 2)
            imagesc(G)
            title("Spectrul canalului G")
            subplot(3, 1, 3)
            imagesc(B)
            title("Spectrul canalului B")
            
            ox = (0:n-1) ./ n;
            oy = (0:m-1) ./ m;
            
            figure
            subplot(3, 1, 1)
            [X, Y] = meshgrid(ox, oy);
            colormap jet
            h = surf(Y, X, R, 'FaceColor','interp');
            set(h,'LineStyle','none')
            set(gca,'Xdir','reverse','Ydir','reverse')
            title("Spectrul canalului R")
            
            subplot(3, 1, 2)
            [X, Y] = meshgrid(ox, oy);
            colormap jet
            h = surf(X, Y, G, 'FaceColor','interp');
            set(h,'LineStyle','none')
            set(gca,'Xdir','reverse','Ydir','reverse')
            title("Spectrul canalului G")
            
            subplot(3, 1, 3)
            [X, Y] = meshgrid(ox, oy);
            colormap jet
            h = surf(X, Y, B, 'FaceColor','interp');
            set(h,'LineStyle','none')
            set(gca,'Xdir','reverse','Ydir','reverse')
            title("Spectrul canalului B")
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
            title("Magnitudine complexă a spectrului FFT")
            xlabel("Frecventa normalizata")
            if strcmp(oy, 'abs')
                ylabel("|fft(X)|")
            elseif strcmp(oy, 'power')
                ylabel("Putere Spectrală [dB]")
            end
        end
        
        function plot_1d_segmente(bucatiTransf, Transf, Canal)
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
            if strcmp(Canal, "") == 0
                title("Spectru transformata " + Transf + ", canal " + Canal)
            else
                title("Spectru transformata " + Transf)
            end
            
            if strcmp(Transf,'Haar2D') == 1 | strcmp(Transf, 'WHT2D') == 1 | strcmp(Transf, 'TKL2D' == 1)
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
        
        function plot_1d_segmente_hwt(bucatiTransf, Transf, lungSegment)
            numSegments = length(bucatiTransf) / lungSegment;
            
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
            oy = 0:size(orig, 1)-1; % y-axis corresponds to the number of rows
            ox = 0:size(orig, 2)-1; % x-axis corresponds to the number of columns
            [X, Y] = meshgrid(ox, oy); % Create a grid that matches the image dimensions
            
            % Calculate the absolute difference error
            er = abs(orig-rec);
            
            % Check if the image is grayscale or color
            if size(orig, 3) == 1
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
                er1 = abs(orig(:, :, 1) - rec(:, :, 1));
                er2 = abs(orig(:, :, 2) - rec(:, :, 2));
                er3 = abs(orig(:, :, 3) - rec(:, :, 3));
                
                figure;
                colormap jet;
                
                % Plot error for the Red channel
                subplot(3, 1, 1);
                h = surf(X, Y, double(er1), 'FaceColor', 'interp');
                set(h, 'LineStyle', 'none');
                xlabel('Index Coloana');
                ylabel('Index Linie');
                zlabel('Eroare Rosu');
                title('Eroarea 2D canalul R');
                colorbar;

                textPositionX = max(get(gca, 'XLim')) * 0.8;
                textPositionY = max(get(gca, 'YLim')) * 0.1;
                normaErorii = sum(er1(:));
                text(textPositionX, textPositionY, max(er1(:)), ['Norma erorii: ' num2str(normaErorii)]);
                
                % Plot error for the Green channel
                subplot(3, 1, 2);
                h = surf(X, Y, double(er2), 'FaceColor', 'interp');
                set(h, 'LineStyle', 'none');
                xlabel('Index Coloana');
                ylabel('Index Linie');
                zlabel('Eroare Verde');
                title('Eroarea 2D canalul G');
                colorbar;

                textPositionX = max(get(gca, 'XLim')) * 0.8;
                textPositionY = max(get(gca, 'YLim')) * 0.1;
                normaErorii = sum(er2(:));
                text(textPositionX, textPositionY, max(er2(:)), ['Norma erorii: ' num2str(normaErorii)]);
                
                % Plot error for the Blue channel
                subplot(3, 1, 3);
                h = surf(X, Y, double(er3), 'FaceColor', 'interp');
                set(h, 'LineStyle', 'none');
                xlabel('Index Coloana');
                ylabel('Index Linie');
                zlabel('Eroare Albastru');
                title('Eroarea 2D canalul B');
                colorbar;

                textPositionX = max(get(gca, 'XLim')) * 0.8;
                textPositionY = max(get(gca, 'YLim')) * 0.1;
                normaErorii = sum(er3(:));
                text(textPositionX, textPositionY, max(er3(:)), ['Norma erorii: ' num2str(normaErorii)]);
            end
        end
        
        function [energie, tabel] = proc_energie_1d(coef, Fs, Transf)
            if strcmp(Transf, 'Fourier')
                coef = coef(1:length(coef)/2+1);
                coef(2:end-1) = 2*coef(2:end-1);
            else
                coef = cat(1, coef{:});
            end
            
            energie = abs(coef).^2;
            energie = sort(energie, 'descend');
            energie_totala = sum(energie);
            
            procente = [0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.99];
            indici = zeros(12, 1);
            
            suma = 0;
            j = 0;
            
            if strcmp(Transf, 'Fourier')
                [f, ~] = utils.plot_1d(coef, Fs, 'power');
            else
                figure
                plot(coef);
            end
            
            for i = 1 : length(procente)
                while (suma <= procente(i) * energie_totala) & j < length(energie)
                    j = j + 1;
                    suma = suma + energie(j);
                end
                indici(i) = j-1;
            end
            
            procente_coef = zeros(length(indici), 1);
            
            for i = 1 : length(indici)
                procente_coef(i) = indici(i)/length(energie) * 100;
                if strcmp(Transf, 'Fourier')
                    xline(f(indici(i)), 'r--', [num2str(procente(i)*100) '% - Procent coef.: ' num2str(procente_coef(i)) '%'], 'LineWidth', 1);
                else
                    xline(indici(i), 'r--', [num2str(procente(i)*100) '% - Procent coef.: ' num2str(procente_coef(i)) '%'], 'LineWidth', 1);
                end
            end
            title('Procente coeficienti/energie')

            tabel = [procente.*100; procente_coef'];
        end

        function [energie, tabel] = proc_energie_2d(coef, Transf)
            if strcmp(Transf, 'TKL')
                if size(coef, 2) == 1
                    coefV{1} = cat(1, coef{1}{:});
                else
                    for i = 1 : size(coef, 2)
                        coefV{i} = cat(1, coef{i}{:});
                    end
                end
            else 
                if size(coef, 2) == 1
                    coefV{1} = reshape(coef{1},[],1);
                else
                    for i = 1 : size(coef, 2)
                        coefV{i} = reshape(coef{i}, [], 1);
                    end
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
            
            ch = size(energie, 2);

            if ch == 3
                canale = [" - canal R", " - canal G", " - canal B"];
            else
                canale = [""];
            end

            procente_coef = {};
            for i = 1 : ch
                figure
                plot(energie{i})
                for j = 1 : length(indici{i})
                    procente_coef{i}(j) = indici{i}(j)/length(coefV{i}) * 100;
                    xline(indici{i}(j), 'r--', [num2str(procente(j)*100) '% - Procent coef.: ' num2str(procente_coef{i}(j)) '%'], 'LineWidth', 1);
                end
                canale(i)
                title(['Procente coeficienti/energie' canale(i)]);
            end

            if size(coef, 2) == 1
                 tabel = [procente.*100; procente_coef{1}];
            else
                 tabel = [procente.*100; procente_coef{1}; procente_coef{2}; procente_coef{3}];
            end
        end

        function panta = regresie_energie(tabel, Transf, color)
            canale = ['R', 'G', 'B'];
            x = tabel(1, :)';

            for i = 2 : size(tabel, 1)
                y{i-1} = tabel(i, :)';
            end

            figure
            for i = 1 : size(y, 2)
                format long
                X = [ones(length(x),1) x];
                size(X)
                b = X\y{i};
                size(b)
                panta = b(2);
                yCalc1 = X*b;
                if size(y, 2) == 3
                    subplot(3, 1, i)
                end
                scatter(x,y{i}, [], color)
                hold on
                plot(x,yCalc1, color)
                xlabel('% energie din energia totala')
                ylabel('% coeficienti')
                if size(y, 2) == 3
                    title(['Dreapta de regresie a compresiei metodei ', Transf, ' - canal ', canale(i)])
                else
                    title(['Dreapta de regresie a compresiei metodei ', Transf])
                end
                grid on
            end
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
        
    end
end