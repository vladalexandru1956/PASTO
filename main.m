clc; clear;
import functii.*


%% Citire fisier
% [file, path] = uigetfile;
% filePath = fullfile(path, file);

filePath = 'D:\pasto_prj\PASTO\Image_1\Lena_standard_bw.bmp'
%% FFT 1D

%[orig, y, Fs] = fft1d(filePath);
%z = inv_fft1d(y, Fs);
%plot_eroare_1d(orig, z, Fs);
%[fftizat_jum, energie, proc_coef] = proc_energie(y, Fs);


%% FFT 2D
% [orig, fftizata] = fft2d(filePath);
% z = inv_fft2d(fftizata);
% plot_eroare_2d(orig, z)


%% TKL 1D
% [orig, y, D, Vm, xM, Fs] = tkl1d(filePath);
% z = inv_tkl1d(y, Vm, xM);
% proc_energie_klt(cat(1, D{:}))

%% TKL 2D
% [orig, y, Vm, xM, xdim, ydim] = tkl2d(filePath);
% z = inv_tkl2d(y, Vm, xM, xdim, ydim);
% figure
% imagesc(abs(z))
% if(size(z, 3) ~= 3)
%     colormap('gray')
% end
% figure
% imagesc(abs(orig))
% if(size(orig, 3) ~= 3)
%     colormap('gray')
% end

%% Haar 1D
%[orig, y, huri, r, Fs] = haar1d(filePath, 10000);
%z = inv_haar1d(huri, r);
%norm(orig-z(1:size(orig)))

%% Haar 2D
%[orig, coef, huri, r, huri_col, r_col] = haar2d(filePath);
%z = inv_haar2d(huri, r, huri_col, r_col);
%imagesc(uint8(z))


%% Wht 1D
% [audio, y, Fs, walshMatrix]  = wht1d(filePath, 1024);
% z = inv_wht1d(y, walshMatrix);
% z = z(1:length(audio));

% norm(audio-z)


%% Wht 2D
[y, orig, walshMatrix_col, walshMatrix_row, xdim_padded, ydim_padded, xdim_orig, ydim_orig] = wht2d(filePath);
z = inv_wht2d(y, walshMatrix_col, walshMatrix_row, xdim_padded, ydim_padded, xdim_orig, ydim_orig);
% z = z';
%plot orig image
% figure
% imagesc(abs(orig))
% colormap('gray')
% plot reconstructed image
% figure
% imagesc(abs(z))
% colormap('gray')
% plot_eroare_2d(orig, z)
[energie, coefV, procente_coef, indici] = proc_energie_2d(y);

% [orig, coef, huri, r, huri_col, r_col] = haar2d(filePath);
% z = inv_haar2d(huri, r, huri_col, r_col);




