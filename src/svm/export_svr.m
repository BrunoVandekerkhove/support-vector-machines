function export_svr()
    h = gcf; %findobj(gcf,'Tag','Fig1');
    set(h, 'PaperUnits','centimeters');
    set(h, 'Units','centimeters');
    pos=get(h,'Position');
    set(h, 'PaperSize', [pos(3) pos(4)]);
    set(h, 'PaperPositionMode', 'manual');
    set(h, 'PaperPosition',[0 0 pos(3) pos(4)]);
    h.OuterPosition = h.InnerPosition;
    print('test', '-dpdf', '-noui', '-fillpage')
end