function export_pdf(h, output_name, width, height)
%EXPORT_PDF Exports the given figure to a pdf file.
    if nargin > 2
        set(h, 'Position', [0 0 width height])
    end
    set(h, 'PaperUnits','centimeters');
    set(h, 'Units','centimeters');
    pos = get(h,'Position');
    set(h, 'PaperSize', [pos(3) pos(4)]);
    set(h, 'PaperPositionMode', 'manual');
    set(h, 'PaperPosition',[0 0 pos(3) pos(4)]);
    print('-dpdf', strcat('figures/', output_name));
end