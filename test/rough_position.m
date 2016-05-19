for step = 0: 100    
    imgUv = imread(strcat('pictures/', num2str(step, '%04d'), '_Uv.jpg'));
    imgUvBin = im2bw(imgUv, 0.03);
    imshow(imgUvBin)
    centerProps = regionprops(imgUvBin, 'Centroid', 'Area', 'Eccentricity');

    center_x = [];
    center_y = [];

    rect_x = [];
    rect_y = [];

    for i = 1: length(centerProps)
        if centerProps(i).Area >= 260 && centerProps(i).Area <= 1500
            if centerProps(i).Eccentricity < 0.75
                center_x = cat(1, center_x, centerProps(i).Centroid(1));
                center_y = cat(1, center_y, centerProps(i).Centroid(2));
            else
                rect_x = cat(1, rect_x, centerProps(i).Centroid(1));
                rect_y = cat(1, rect_y, centerProps(i).Centroid(2));
            end
        end
    end

    orien = []; 

    for i = 1: length(center_x)
        minVal = 10000;
        index = 1;
        for j = 1: length(rect_x)
            dist2 = (rect_x(j) - center_x(i))^2 + (rect_y(j) - center_y(i))^2;
            if dist2 < minVal
                minVal = dist2;
                index = j;
            end
        end
        orien = cat(1, orien, atan2(rect_y(index) - center_y(i), rect_x(index) - center_x(i)));
    end

    f = fopen(strcat('roughposition/step_', num2str(step, '%04d')), 'w');
    for i = 1: length(center_x)
        fprintf(f, '%12f %12f %12f\n', center_x(i), center_y(i), orien(i));
    end
    fclose(f);
end










