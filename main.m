clc; clear;
import tfft.*
import utils.*
import tkl.*
import haar.*
import wht.*


%% Citire fisier
 [file, path] = uigetfile;
 filePath = fullfile(path, file);

%% FFT 1D

% [orig, y, Fs] = fft1d(filePath);
% z = inv_fft1d(y, Fs);
% plot_eroare_1d(orig, z, Fs);
% [~, tabel_fft1d] = proc_energie_1d(y, Fs, 'Fourier');
% panta_fft1d = regresie_energie(tabel_fft1d, 'Fourier', 'r');


%% FFT 2D
%[orig, fftizata, coef_fft2d] = fft2d(filePath);
%z = inv_fft2d(fftizata);
%plot_eroare_2d(double(orig), double(z));

%figure
%imagesc(abs(uint8(z)))
%if(size(z, 3) ~= 3)
%   colormap('gray')
%end
%title("Imaginea reconstruita FFT2D")

%[~, tabel_fft2d] = proc_energie_2d(coef_fft2d, 'FFT2D');
%panta_fft2d = regresie_energie(tabel_fft2d, 'FFT2D', 'r');


%% TKL 1D
%[orig, coef, D, Vm, xM, Fs] = tkl1d(filePath);
%z = inv_tkl1d(coef, Vm, xM);
%T = 1 / Fs;
            
%figure
%t = (0:length(z)-1)*T;
%plot(t,z)
%title("Semnalul audio reconstruit")
%xlabel("t (seconds)")
%ylabel("y(t)")
%plot_eroare_1d(orig, z, Fs);
%[~, tabel_tkl1d] = proc_energie_1d(coef, 0, 'TKL');
%panta_tkl1d = regresie_energie(tabel_tkl1d, 'TKL1D', 'g');

%% TKL 2D
% [orig, coef_tkl2d, Vm, xM, xdim, ydim] = tkl2d(filePath);
% z = inv_tkl2d(coef_tkl2d, Vm, xM, xdim, ydim);
% figure
% imagesc(uint8(z))
% if(size(z, 3) ~= 3)
%     colormap('gray')
% end
% title("Imagine reconstruita TKL2D")
% plot_eroare_2d(double(orig), double(z))
% [~, tabel_tkl2d] = proc_energie_2d(coef_tkl2d, 'TKL');
% panta_tkl2d = regresie_energie(tabel_tkl2d, 'TKL2D', 'g');

%% Haar 1D
%[orig, y, huri, r, Fs] = haar1d(filePath, 10000);
% z = inv_haar1d(huri, r);
% % T = 1 / Fs;
% % figure
% % t = (0:length(z)-1)*T;
% % plot(t,z)
% % title("Semnalul audio reconstruit")
% % xlabel("t (seconds)")
% % ylabel("y(t)")
% plot_eroare_1d(orig, z(1:size(orig)), Fs)
% norm(orig-z(1:size(orig)))
%[~, tabel_haar1d] = proc_energie_1d(y, 0, 'Haar1D');
%panta_haar1d = regresie_energie(tabel_haar1d, 'Haar1D', 'b')

%% Haar 2D
[orig, coef, huri, r, huri_col, r_col] = haar2d(filePath);
z = inv_haar2d(huri, r, huri_col);
plot_eroare_2d(double(orig), double(z))
[~, tabel_haar2d] = proc_energie_2d(coef, 'Haar2D');
figure
imagesc(uint8(z))
if(size(z, 3) ~= 3)
   colormap('gray')
end
title('Imaginea recosntruita Haar2D')
panta_haar2d = regresie_energie(tabel_haar2d, 'Haar2D', 'b');

%% Wht 1D
%[audio, y, Fs, walshMatrix]  = wht1d(filePath, 1024);
%z = inv_wht1d(y, walshMatrix);
%T = 1 / Fs;
%figure
%t = (0:length(z(1:length(audio)))-1)*T;
%plot(t,z(1:length(audio)))
%title("Semnalul audio reconstruit")
%xlabel("t (seconds)")
%ylabel("y(t)")
%z = z(1:length(audio));
%[~, tabel_wht1d] = proc_energie_1d(y, 0, 'WHT1D');
%panta_wht1d = regresie_energie(tabel_wht1d, 'WHT1D', 'm')

%norm(audio-z)


%% Wht 2D
% [y, orig, walshMatrix_col, walshMatrix_row, xdim_padded, ydim_padded, xdim_orig, ydim_orig] = wht2d(filePath);
% z = inv_wht2d(y, walshMatrix_col, walshMatrix_row, xdim_padded, ydim_padded, xdim_orig, ydim_orig);
%  figure
%  imagesc(abs(uint8(z)))
%  if(size(z, 3) ~= 3)
%      colormap('gray')
%  end
%  title('Imaginea recosntruita TWHT2D')
% plot_eroare_2d(double(orig), double(z));
% [energie, tabel_wht2d] = proc_energie_2d(y, 'WHT2D');
% panta_wht2d = regresie_energie(tabel_wht2d, 'WHT2D', 'm');

%% Toate regresiile pe acelasi grafic
%trebuie comentat figure-ul si partea de titlu din functia
%utils.regresie_energie

% figure
% panta_fft1d = regresie_energie(tabel_fft1d, 'FFT1D', 'r');
% panta_tkl1d = regresie_energie(tabel_tkl1d, 'TKL1D', 'g');
% panta_haar1d = regresie_energie(tabel_haar1d, 'Haar1D', 'b');
% panta_wht1d = regresie_energie(tabel_wht1d, 'WHT1D', 'm');
% title('Drepte de regresie ale compresiei fiecarei transformate pentru un semnal 1D')
% legend('FFT1D', 'procente_{fft1d}', 'TKL1D', 'procente_{tkl1d}', 'Haar1D', 'procente_{haar1d}', 'WHT1D', 'procente_{wht1d}', 'Location', 'northwest');
% lines = {['Panta FFT1D: ' num2str(panta_fft1d)], ['PantaTKL1D: ' num2str(panta_tkl1d)], ['Panta Haar1D: ' num2str(panta_haar1d)], ['Panta WHT1D: ' num2str(panta_wht1d)]};
% gtext(lines)

% figure
% panta_fft2d = regresie_energie(tabel_fft2d, 'FFT2D', 'r');
% panta_tkl2d = regresie_energie(tabel_tkl2d, 'TKL2D', 'g');
% panta_haar2d = regresie_energie(tabel_haar2d, 'Haar2D', 'b');
% panta_wht2d = regresie_energie(tabel_wht2d, 'WHT2D', 'm');
% title('Drepte de regresie ale compresiei fiecarei transformate pentru un semnal 2D')
% legend('FFT2D', 'procente_{fft2d}', 'TKL2D', 'procente_{tkl2d}', 'Haar2D', 'procente_{haar2d}', 'WHT2D', 'procente_{wht2d}', 'Location', 'northwest');
% lines = {['Panta FFT2D: ' num2str(panta_fft2d)], ['PantaTKL2D: ' num2str(panta_tkl2d)], ['Panta Haar2D: ' num2str(panta_haar2d)], ['Panta WHT2D: ' num2str(panta_wht2d)]};
% gtext(lines)
