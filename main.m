import functii.*

[file, path] = uigetfile;
filePath = fullfile(path, file);

[orig, y, Fs] = fft1d(filePath);
z = inv_fft1d(y, Fs);
plot_eroare_1d(orig, z, Fs);
[fftizat_jum, energie, proc_coef] = proc_energie(y, Fs);


% [orig, fftizata] = fft2d(filePath);
% z = inv_fft2d(fftizata);
% plot_eroare_2d(orig, z)


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

% T = 1/Fs;

% figure
%             t = (0:length(z)-1)*T;
%             plot(t,z, 'r')
%             hold on
%             plot(t,orig, 'b')
%             xlabel("t (seconds)")
%             ylabel("z(t)")

            % figure
            % t = (0:length(orig)-1)*T;
           
            % xlabel("t (seconds)")
            % ylabel("orig(t)")