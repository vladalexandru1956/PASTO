import functii.*

[file, path] = uigetfile;
filePath = fullfile(path, file);

%[orig, y, Fs] = fft1d(filePath);
%z = inv_fft1d(y, Fs);
%plot_eroare_1d(orig, z, Fs);
%[fftizat_jum, energie, proc_coef] = proc_energie(y, Fs);

 % 
 % [orig, fftizata] = fft2d(filePath);
 % z = inv_fft2d(fftizata);
 % plot_eroare_2d(orig, z)

%[orig, y, huri, r, Fs] = haar1d(filePath, 4096);
%z = inv_haar1d(huri, r);
%norm(orig-z(1:size(orig,1)))

[orig, coef, huri, r, huri_col, r_col] = haar2d(filePath);
[z, X_inter, x_inter] = inv_haar2d(huri, r, huri_col, r_col); 
figure
imagesc(uint8(z))
figure
imagesc(uint8(orig))
%colormap('gray')
plot_eroare_2d(orig, z)



% [orig, y, D, Vm, xM, Fs] = tkl1d(filePath);
% z = inv_tkl1d(y, Vm, xM);
% proc_energie_klt(cat(1, D{:}))
% 
% [orig, y, Vm, xM, xdim, ydim] = tkl2d(filePath);
% z = inv_tkl2d(y, Vm, xM, xdim, ydim);
% 
% figure
% imagesc(abs(z))
% 
% if(size(z, 3) ~= 3)
%     colormap('gray')
% end
% 
% figure
% imagesc(abs(orig))
% 
% if(size(orig, 3) ~= 3)
%     colormap('gray')
% end

% C = rand(500, 1).^2;
% [y, Vm, xM, x_centr, p_x, R, D, V] = proc_tkl1d(C);
% z = inv_tkl1d(y, Vm, xM);

%[orig, coef, huri, r, huri_col, r_col] = haar2d(filePath);
%z = inv_haar2d(huri, r, huri_col, r_col);
%imagesc(uint8(z))


%[orig, y, huri, r, Fs] = haar1d(filePath, 10000);
%z = inv_haar1d(huri, r);
%norm(orig-z(1:size(orig)))

% 
% % T = 1/Fs;
% 
% % figure
%  %            t = (0:length(z(1:length(orig)))-1)*T;
%   %           plot(t,z(1:length(orig)), 'r')
%   %           hold on
% 
%              %xlabel("t (seconds)")
%              ylabel("z(t)")
% 
%          %    figure
%          %    t = (0:length(orig)-1)*T;
%              plot(t,orig, 'b')
%              xlabel("t (seconds)")
%              ylabel("orig(t)")
