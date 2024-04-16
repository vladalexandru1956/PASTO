classdef    tfft
    methods     ( Static = true )
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
            
            utils.plot_1d(audio_fft, Fs, 'abs');
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
        
        function [imagine, imagine_fft, coef] = fft2d(image)
            imagine = imread(image);
            [~, ~, ch] = size(imagine);
            
            imagine_fft = fft2(double(imagine));
            
            if(ch == 3)
                utils.plot_rgb(imagine_fft);
                coef{1} = imagine_fft(:, :, 1);
                coef{2} = imagine_fft(:, :, 2);
                coef{3} = imagine_fft(:, :, 3);
            else
                utils.plot_gray(imagine_fft);
                coef{1} = imagine_fft;
            end

        end
        
        function y = inv_fft2d(x)
       %     y = uint8(ifft2(x));
            y = real(ifft2(x));
          
        end
    end
end