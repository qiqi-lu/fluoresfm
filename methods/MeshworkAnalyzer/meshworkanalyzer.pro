pro initializeNODE, wwidget
  common display_info, def_w, wxsz,wysz, mousedown, infoinitial, frontendwindow, frontendzoom
  common meshwork_mask, ImageROI_ptrarr_ptrarr
  common display, wxszx, wyszx, mainwindow, zoomcoord, autoscale, screenmode
  Common meshwork_dataset, properties, image_set, filamentmask, nodemask, node_table, pore_table, filament_table, angle_table, aster_table


  wxsz = 1024. & wysz = 1024.
  xydsz = [1024,1024]
  def_w = !D.WINDOW
  frontendwindow= !D.WINDOW
  mousedown = 0
  mainwindow = !D.WINDOW

  frontendzoom = [0,0,1023,1023]
  boxcolor = !D.N_colors-10
  info = {image: dblarr(2,2), $
    wid:frontendwindow, $
    drawID: widget_info(wWidget,find_by_uname='WID_DRAW_MAIN'), $
    pixID:-1, $
    xsize:wxsz, $
    ysize:wysz, $
    sx:-1,$
    sy:-1,$
    boxColor: boxColor}
  infoinitial = info
  widget_control, wWidget, set_uvalue =info, /no_copy

  print, 'initialized..'
  device,decompose=0
  cgloadct, 0, /reverse

  properties = {datafile: '',timestamp: 0.,xpixels:512.,ypixels:512.,frames:1.}
  widget_control,widget_info(wwidget,find_by_uname='WID_DROPLIST_COLOR'),set_droplist_select=5

  image_transform_parameters = [0,0,0.]
end

pro WID_BASE_node_event, Event
  common display_info, def_w, wxsz,wysz, mousedown, infoinitial, frontendwindow, frontendzoom
  Common meshwork_dataset, properties, image_set, filamentmask, nodemask, node_table, pore_table, filament_table, angle_table, aster_table

  wTarget = (widget_info(Event.id,/NAME) eq 'TREE' ?  $
    widget_info(Event.id, /tree_root) : event.id)
  wWidget =  Event.top

  wxsz = 1024
  wysz = 1024

  ; colordroplist=['Red Temperature # 3','Grayscale # 0','Rainbow # 13','Inverse Grayscale','Rainbow #39','Brewer Red-Blue 22']
  colordroplist=[3,0,13,0,39,22]
  displaylist = ['SMLM','OFT','Flmnt','ROI','Overlay-mask/smlm','Overlay-oft/SMLM','Overlay-Flmnt/SMLM','Asters/SMLM','Asters/Flmnt']
  matlabmode = widget_info(Widget_Info(wWidget, FIND_BY_UNAME='WID_DROPLIST_MATLAB'),/droplist_select)
  ;  widget_control, Widget_Info(wWidget, FIND_BY_UNAME='WID_SLIDER_FRAME'),get_value=frame
  colorchoice = widget_info(Widget_Info(wWidget, FIND_BY_UNAME='WID_DROPLIST_COLOR'),/droplist_select)
  headermode = widget_info(Widget_Info(wWidget, FIND_BY_UNAME='WID_DROPLIST_HEADER'),/droplist_select)
  displaymode = widget_info(Widget_Info(wWidget, FIND_BY_UNAME='WID_DROPLIST_DISPLAY'),/droplist_select)

  if matlabmode eq 1 then widget_control,Widget_Info(wWidget, FIND_BY_UNAME='WID_TEXT_MATLAB'),get_value=matlabregistry else matlabregistry = !null
  widget_control,Widget_Info(wWidget, FIND_BY_UNAME='WID_SLIDER_OFTRADIUS'),get_value=oftradius
  widget_control,Widget_Info(wWidget, FIND_BY_UNAME='WID_SLIDER_OFTSECTOR'),get_value=oftsector

  widget_control,Widget_Info(wWidget, FIND_BY_UNAME='WID_SLIDER_OFTTHRESHOLD'),get_value=oftthreshold
  widget_control,Widget_Info(wWidget, FIND_BY_UNAME='WID_SLIDER_OFTTHRESHOLDMULTIPLIER'),get_value=oftthresholdmultiplier
  widget_control,Widget_Info(wWidget, FIND_BY_UNAME='WID_SLIDER_SMLMPIXELSIZE'),get_value=smlmpixelsize
  smlmpixelsizeum = smlmpixelsize/1000.

  case wTarget of
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_EXIT'):           widget_control, event.top,/destroy
    Widget_Info(wWidget, FIND_BY_UNAME='WID_DRAW_MAIN'): meshwork_drawevents,event

    ;===display
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_CLEARSCREEN'): cgerase,'white'
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_CLEARSCREEN2'):cgerase,'white'
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_COPYSCREEN'):begin
      presentimage=tvrd(true=1)
      cgwindow, wxsize = 1024., wysize = 1024.
      cgimage, presentimage,position=[0.,0.,1.,1.],/addcmd, /keep_aspect_ratio,/interpolate
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_RESETZOOM'):begin
      meshwork_display,event,/reset
      meshwork_display_refresh, event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_RESETZOOM2'):begin
      meshwork_display,event,/reset
      meshwork_display_refresh, event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_REFRESH'):  meshwork_display_refresh, event
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_REFRESH2'):  meshwork_display_refresh, event
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_ZOOMOUT2X'):begin
      meshwork_display,event,/out2x
      meshwork_display_refresh, event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_QUARTERVIEW'):begin
      meshwork_display, event, /smlm, /q2,/quatre,/noerase
      meshwork_display, event, /skeleton, /q3,/quatre,/noerase
      meshwork_display, event, /ofoverlay, /q1,/quatre,/noerase
      meshwork_display, event, /oft, /q4, /quatre,/noerase
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_QUARTERVIEW2'):begin
      meshwork_display, event, /smlm, /q2,/quatre,/noerase
      meshwork_display, event, /skeleton, /q3,/quatre,/noerase
      meshwork_display, event, /lft, /q1,/quatre,/noerase
      meshwork_display, event, /oft, /q4, /quatre,/noerase
    end
    widget_info(wWidget, FIND_BY_UNAME='WID_DROPLIST_COLOR'): meshwork_color_refresh,event
    widget_info(wWidget, FIND_BY_UNAME='WID_BUTTON_SCALEBAR'): meshwork_scalebar, event

    ;===I/O
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_LOADTIFF'):begin
      meshwork_loadtiff, event
      meshwork_display,event, /smlm
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_LOADOFT'):begin
      meshwork_loadoft, event, image = image
      cgimage, image, /scale,position=[0,0,1.,1.]
      meshwork_imageset_transform, event, updateoft=image
      meshwork_display_refresh, event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_LOADFILAMENTMATLAB'): meshwork_loadoft, event, /watershed


    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_LOADSAV'):begin
      meshwork_io,event,/loadsav
      meshwork_display,event,/reset
      meshwork_display_refresh, event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_SAVESCREEN'):   meshwork_savescreentiff, event
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_SAVESAV'):meshwork_io,event,/savesav
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_EXPORTCROP'):meshwork_io,event,/savesav,/crop

    ;===enhancement/segmentation
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_SEGMENTATION'):begin
      if size(image_set,/type) eq 0 then return
      imageraw = image_set[*,*,0]
      oldmask = image_set[*,*,4]
      mask = meshwork_segment(imageraw,groupleader = wWidget, /interactive,default=oldmask)
      help, mask
      image_set[*,*,4]  = mask
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_EDITSEGMENTATION'):begin
      if size(image_set,/type) eq 0 then return
      imageraw = image_set[*,*,0]
      oldmask = image_set[*,*,4]
      newmask = meshwork_segment_edit(oldmask, groupleader=wWidget, grayscale = imageraw)
      image_set[*,*,4]  = newmask
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_EDITWATERSHED'):begin
      if size(image_set,/type) eq 0 then return
      imageraw = image_set[*,*,0]
      oldmask = reform(image_set[*,*,2]) eq 0
      newmask = meshwork_segment_edit(oldmask, groupleader=wWidget, grayscale = imageraw)
      image_set[*,*,2]  = meshwork_watershed_relabel(watershedimage=~newmask)
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_SEGMENTATIONASTERS'):begin
      if size(image_set,/type) eq 0 then return
      imageraw = image_set[*,*,0]
      oldmask = image_set[*,*,5]
      mask = meshwork_segment(imageraw,groupleader = wWidget, /interactive,default=oldmask)
      help, mask
      if max(image_set[*,*,4]) gt 0 then image_set[*,*,5]=mask*reform(image_set[*,*,4]) else image_set[*,*,5]  = mask
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_EDITSEGMENTATIONASTERS'):begin
      if size(image_set,/type) eq 0 then return
      imageraw = image_set[*,*,0]
      oldmask = image_set[*,*,5]
      newmask = meshwork_segment_edit(oldmask, groupleader=wWidget, grayscale = imageraw)
      if max(image_set[*,*,4]) gt 0 then image_set[*,*,5]=newmask*reform(image_set[*,*,4]) else image_set[*,*,5]  = newmask
    end



    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_RESETSEGMENTATION'): image_set[*,*,4]  = imageset[*,*,0]*0
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_OFTOTSU'):meshwork_oft, event, /threshold
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_OVERLAYOFTTHRESHOLD'):meshwork_oft, event, /overlay
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_SKELETONSMLM'):meshwork_oft, event, /skeleton
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_HMINIMAOFT'):meshwork_oft, event, /hminima
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_OFTHISTOGRAM'): meshwork_oft, event, /histogram
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_OFTTEST'): meshwork_oft, event, /test
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_OFTFULL'): begin
      meshwork_display,event,/reset
      meshwork_display_refresh, event
      meshwork_oft,event,/full
      widget_control,widget_Info(wWidget, FIND_BY_UNAME='WID_DROPLIST_DISPLAY'),set_droplist_select = 5
      meshwork_display_refresh, event
      meshwork_oft, event, /threshold
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_HMINIMAOFTWATERSHED'):meshwork_oft, event, /watershed
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_EXTRACTFILAMENT'):begin
      meshwork_display,event,/reset
      meshwork_display, event, /smlm
      widget_control,Widget_Info(wWidget, FIND_BY_UNAME='WID_DROPLIST_DISPLAY'),set_droplist_select=0
      if max(image_set[*,*,4]) eq 0 then meshwork_oft, event, /watershed,/use,/nomask else meshwork_oft, event, /watershed,/use
    end

    ;===Filament tracing
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_UPDATETRACING'): meshwork_processing,event, /process,/verbose
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_UPDATEANGLES'):  meshwork_angles, event, /all
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_UPDATEALL'):  begin
      meshwork_processing,event, /process,/verbose
      meshwork_angles, event, /all
      meshwork_process_asters, event, /update
    end


    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_PLOTPORES'): meshwork_display, event, /pores
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_PLOTNODES'): meshwork_display, event, /nodes
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_PLOTFILAMENTS'): meshwork_display, event,/filaments
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_PLOTNODESID'): meshwork_display, event, /idnode
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_PLOTFILAMENTSID'):meshwork_display, event,/idfilaments
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_PLOTPORESID'): meshwork_display, event, /idpores
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_SHOWANGLES'): meshwork_display, event, /angles



    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_PINPOINTNODE'): begin
      selectedrow = (widget_info(widget_info(event.top,find_by_uname='WID_TABLE_NODES'),/table_select))[1,*]
      sel = selectedrow(uniq(selectedrow))
      meshwork_display, event, /nodes , selection= sel, /pinpoint;, /xygoto
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_PINPOINTFILAMENTS'):begin
      selectedrow = (widget_info(widget_info(event.top,find_by_uname='WID_TABLE_FILAMENTS'),/table_select))[1,*]
      sel = selectedrow(uniq(selectedrow))
      ;print, sel
      meshwork_display, event, /filaments , selection= sel, /pinpoint
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_PINPOINTPORES'):begin
      selectedrow = (widget_info(widget_info(event.top,find_by_uname='WID_TABLE_PORES'),/table_select))[1,*]
      sel = selectedrow(uniq(selectedrow))
      meshwork_display, event, /pores , selection= sel, /pinpoint
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_PINPOINTANGLES'):begin
      selectedrow = (widget_info(widget_info(event.top,find_by_uname='WID_TABLE_ANGLES'),/table_select))[1,*]
      sel = selectedrow(uniq(selectedrow))
      ;widget_control,widget_info(event.top,find_by_uname='WID_TABLE_ANGLES'),get_value=angle_table
      meshwork_display, event, /angles , selection= sel, /pinpoint
    end

    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_PINPOINTASTERS'): begin
      selectedrow = (widget_info(widget_info(event.top,find_by_uname='WID_TABLE_ASTERS'),/table_select))[1,*]
      sel = selectedrow(uniq(selectedrow))
      meshwork_asters, event, /show, selection=sel
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_FITNODES'): begin
      selectedrow = (widget_info(widget_info(event.top,find_by_uname='WID_TABLE_NODES'),/table_select))[1,*]
      sel = selectedrow(uniq(selectedrow))
      meshwork_angles,event, selection=sel , /show
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_FITFILAMENTS'):begin
      selectedrow = (widget_info(widget_info(event.top,find_by_uname='WID_TABLE_FILAMENTS'),/table_select))[1,*]
      sel = selectedrow(uniq(selectedrow))
      ;print, sel
      meshwork_filaments, event, selection=sel , /show
    end


    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_GOTOANGLES'):begin
      selectedrow = (widget_info(widget_info(event.top,find_by_uname='WID_TABLE_ANGLES'),/table_select))[1,*]
      sel = selectedrow(uniq(selectedrow))
      widget_control,widget_info(event.top,find_by_uname='WID_TABLE_ANGLES'),get_value=angle_table
      meshwork_goto, event, x=angle_table[2,sel[0]] , y=angle_table[3,sel[0]]
      meshwork_display_refresh,event
      meshwork_angles,event, selection=angle_table[1,sel[0]], /show,/idnode
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_GOTONODE'): begin
      selectedrow = (widget_info(widget_info(event.top,find_by_uname='WID_TABLE_NODES'),/table_select))[1,*]
      sel = selectedrow(uniq(selectedrow))
      widget_control,widget_info(event.top,find_by_uname='WID_TABLE_NODES'),get_value=node_table
      meshwork_goto, event, x=node_table[1,sel[0]] , y=node_table[2,sel[0]]
      meshwork_display_refresh,event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_GOTOPORE'):begin
      selectedrow = (widget_info(widget_info(event.top,find_by_uname='WID_TABLE_PORES'),/table_select))[1,*]
      sel = selectedrow(uniq(selectedrow))
      widget_control,widget_info(event.top,find_by_uname='WID_TABLE_PORES'),get_value=pore_table
      meshwork_goto, event, x=pore_table[1,sel[0]] , y=pore_table[2,sel[0]]
      meshwork_display_refresh,event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_GOTOFILAMENT'):begin
      selectedrow = (widget_info(widget_info(event.top,find_by_uname='WID_TABLE_FILAMENTS'),/table_select))[1,*]
      sel = selectedrow(uniq(selectedrow))
      widget_control,widget_info(event.top,find_by_uname='WID_TABLE_FILAMENTS'),get_value=filament_table
      meshwork_goto, event, x=filament_table[4,sel[0]] , y=filament_table[5,sel[0]]
      meshwork_display_refresh,event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_GOTOASTERS'): begin
      selectedrow = (widget_info(widget_info(event.top,find_by_uname='WID_TABLE_ASTERS'),/table_select))[1,*]
      sel = selectedrow(uniq(selectedrow))
      widget_control,widget_info(event.top,find_by_uname='WID_TABLE_ASTERS'),get_value=aster_table
      meshwork_goto, event, x=aster_table[1,sel[0]] , y=aster_table[2,sel[0]]
      meshwork_display_refresh,event
    end
    ;===sort===
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_SORTNODE'): begin
      selectedcol = (widget_info(widget_info(event.top,find_by_uname='WID_TABLE_NODES'),/table_select))[*,1]
      sel = (selectedcol(uniq(selectedrow)))[0]
      if sel lt 0 then return
      if n_elements(node_table) eq 0 then return
      coltosort = reform(node_table[sel,*])
      node_table=node_table[*,reverse(sort(coltosort))]
      widget_control,widget_info(event.top,find_by_uname='WID_TABLE_NODES'),set_value=node_table
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_SORTNODEASCEND'): begin
      selectedcol = (widget_info(widget_info(event.top,find_by_uname='WID_TABLE_NODES'),/table_select))[*,1]
      sel = (selectedcol(uniq(selectedrow)))[0]
      if sel lt 0 then return
      if n_elements(node_table) eq 0 then return
      coltosort = reform(node_table[sel,*])
      node_table=node_table[*,sort(coltosort)]
      widget_control,widget_info(event.top,find_by_uname='WID_TABLE_NODES'),set_value=node_table
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_SORTFILAMENTS'): begin
      selectedcol = (widget_info(widget_info(event.top,find_by_uname='WID_TABLE_FILAMENTS'),/table_select))[*,1]
      sel = (selectedcol(uniq(selectedrow)))[0]
      if sel lt 0 then return
      if n_elements(filament_table) eq 0 then return
      coltosort = reform(filament_table[sel,*])
      filament_table=filament_table[*,reverse(sort(coltosort))]
      widget_control,widget_info(event.top,find_by_uname='WID_TABLE_FILAMENTS'),set_value=filament_table
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_SORTFILAMENTSASCEND'): begin
      selectedcol = (widget_info(widget_info(event.top,find_by_uname='WID_TABLE_FILAMENTS'),/table_select))[*,1]
      sel = (selectedcol(uniq(selectedrow)))[0]
      if sel lt 0 then return
      if n_elements(filament_table) eq 0 then return
      coltosort = reform(filament_table[sel,*])
      filament_table=filament_table[*,sort(coltosort)]
      widget_control,widget_info(event.top,find_by_uname='WID_TABLE_FILAMENTS'),set_value=filament_table
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_SORTPORES'): begin
      selectedcol = (widget_info(widget_info(event.top,find_by_uname='WID_TABLE_PORES'),/table_select))[*,1]
      sel = (selectedcol(uniq(selectedrow)))[0]
      if sel lt 0 then return
      if n_elements(pore_table) eq 0 then return
      coltosort = reform(pore_table[sel,*])
      pore_table=pore_table[*,reverse(sort(coltosort))]
      widget_control,widget_info(event.top,find_by_uname='WID_TABLE_PORES'),set_value=pore_table
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_SORTPORESASCEND'): begin
      selectedcol = (widget_info(widget_info(event.top,find_by_uname='WID_TABLE_PORES'),/table_select))[*,1]
      sel = (selectedcol(uniq(selectedrow)))[0]
      if sel lt 0 then return
      if n_elements(pore_table) eq 0 then return
      coltosort = reform(pore_table[sel,*])
      pore_table=pore_table[*,sort(coltosort)]
      widget_control,widget_info(event.top,find_by_uname='WID_TABLE_PORES'),set_value=pore_table
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_SORTANGLES'): begin
      selectedcol = (widget_info(widget_info(event.top,find_by_uname='WID_TABLE_ANGLES'),/table_select))[*,1]
      sel = (selectedcol(uniq(selectedrow)))[0]
      if sel lt 0 then return
      if n_elements(angle_table) eq 0 then return
      coltosort = reform(angle_table[sel,*])
      angle_table=angle_table[*,reverse(sort(coltosort))]
      widget_control,widget_info(event.top,find_by_uname='WID_TABLE_ANGLES'),set_value=angle_table
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_SORTANGLESASCEND'): begin
      selectedcol = (widget_info(widget_info(event.top,find_by_uname='WID_TABLE_ANGLES'),/table_select))[*,1]
      sel = (selectedcol(uniq(selectedrow)))[0]
      if sel lt 0 then return
      if n_elements(angle_table) eq 0 then return
      coltosort = reform(angle_table[sel,*])
      angle_table=angle_table[*,sort(coltosort)]
      widget_control,widget_info(event.top,find_by_uname='WID_TABLE_ANGLES'),set_value=angle_table
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_SORTASTERS'):begin
      selectedcol = (widget_info(widget_info(event.top,find_by_uname='WID_TABLE_ASTERS'),/table_select))[*,1]
      widget_control,widget_info(event.top,find_by_uname='WID_TABLE_ASTERS'),get_value=aster_table
      sel = (selectedcol(uniq(selectedrow)))[0]
      if sel lt 0 then return
      if n_elements(aster_table) eq 0 then return
      coltosort = reform(aster_table[sel,*])
      aster_table=aster_table[*,reverse(sort(coltosort))]
      widget_control,widget_info(event.top,find_by_uname='WID_TABLE_ASTERS'),set_value=aster_table
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_SORTASTERSASCEND'):begin
      selectedcol = (widget_info(widget_info(event.top,find_by_uname='WID_TABLE_ASTERS'),/table_select))[*,1]
      widget_control,widget_info(event.top,find_by_uname='WID_TABLE_ASTERS'),get_value=aster_table
      sel = (selectedcol(uniq(selectedrow)))[0]
      if sel lt 0 then return
      if n_elements(aster_table) eq 0 then return
      coltosort = reform(aster_table[sel,*])
      aster_table=aster_table[*,sort(coltosort)]
      widget_control,widget_info(event.top,find_by_uname='WID_TABLE_ASTERS'),set_value=aster_table
    end

    ;==export==
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_EXPORTNODE'):meshwork_table,event,/export,/ntable
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_EXPORTFILAMENTS'):meshwork_table,event,/export,/ftable
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_EXPORTPORES'): meshwork_table,event,/export,/ptable
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_EXPORTANGLES'): meshwork_table,event,/export,/atable
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_EXPORTASTERS'): meshwork_table,event,/export,/astertable

    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_STATNODES'): meshwork_table,event,/statistics,/ntable
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_STATFILAMENTS'): meshwork_table,event,/statistics,/ftable
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_STATPORES'): meshwork_table,event,/statistics,/ptable
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_STATANGLES'): meshwork_table,event,/statistics,/atable

    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_TOGGLEPORES'):begin
      selectedrow = (widget_info(widget_info(event.top,find_by_uname='WID_TABLE_PORES'),/table_select))[1,*]
      sel = selectedrow(uniq(selectedrow))
      widget_control, widget_info(event.top,find_by_uname='WID_TABLE_PORES'), get_value=pore_table
      oldval = reform(pore_table[5,sel])
      newval = oldval
      newval[where(oldval eq 0, /null)] = -1
      newval[where(oldval ne 0, /null)] = 0
      pore_table[5,sel] = newval
      widget_control, widget_info(event.top,find_by_uname='WID_TABLE_PORES'), set_value=pore_table
    end

    ; == sensitivity
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_SENSITIVITY_ENHANCEMENT'): meshwork_sensitivity, event,/enhancement
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_SENSITIVITY_SEGMENTATION'): meshwork_sensitivity, event,/segmentation
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_SENSITIVITY_GEOMETRY'): meshwork_sensitivity, event,/geometry

    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_SENSITIVITY_SEGMENTATION_ANALYSIS'):    meshwork_sensitivity_analysis,event, /load, /segmentation, /summary
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_UPDATEALLBATCH'): meshwork_process_batch, event

    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_SUMMARYBATCH'):meshwork_quantify, event, pixelsize=smlmpixelsizeum,/batch
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_SUMMARY'): meshwork_quantify, event, pixelsize=smlmpixelsizeum
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_EXPORTPOOLEDHISTO'):meshwork_quantify, event, /export, pixelsize=smlmpixelsizeum, /batch, noheader=headermode
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_EXPORTPOOLEDDATAPTS'):meshwork_quantify, event, /export, pixelsize=smlmpixelsizeum, /batch,/pooled, noheader=headermode
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_EXPORTCELLHISTO'):meshwork_quantify, event, /export, pixelsize=smlmpixelsizeum, /cell, noheader=headermode


    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_UPDATEASTERS'):meshwork_process_asters, event, /update


    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_STATASTERS'):meshwork_table,event,/statistics,/astertable
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_ASTERSAZIMUTH'):begin
      selectedrow = (widget_info(widget_info(event.top,find_by_uname='WID_TABLE_ASTERS'),/table_select))[1,*]
      sel = selectedrow(uniq(selectedrow))+1
      ;print, sel
      meshwork_azimuth,event, selection=[sel],/image,/verbose,/blank;,/oft
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_MONTAGEASTERS'): meshwork_asters,event,/montage

    else:
  endcase
end

pro MeshworkAnalyzer, GROUP_LEADER=wGroup, _EXTRA=_VWBExtra_, image=im,modal=modal
  common display_info, def_w, wxsz,wysz, mousedown, infoinitial, frontendwindow, frontendzoom
  Common meshwork_dataset, properties, image_set, filamentmask, nodemask, node_table, pore_table, filament_table, angle_table, aster_table

  wxsz = 1024
  wysz = 1024

  xoffset=0.
  yoffset=0.
  colordroplist=['Red Temperature # 3','Grayscale # 0','Rainbow # 13','Inverse Grayscale','Rainbow #39','Inverse Blue']
  ;filtrackzoom = [0,0,1023,1023]
  displaylist = ['SMLM','OFT','Flmnt','ROI','Overlay-mask/smlm','Overlay-oft/SMLM','Overlay-Flmnt/SMLM','Asters/SMLM','Asters/Flmnt']
  detectionmodelist = ['Use Mask if available','Entire image']

  WID_BASE_node = Widget_Base( GROUP_LEADER=wGroup,  $
    UNAME='WID_BASE_NODE' ,XOFFSET=0 ,YOFFSET=0  $
    ,SCR_XSIZE=wxsz+600 ,SCR_YSIZE=wysz+90  $
    ,TITLE='Meshwork Analyzer release 1 (c) 2017-8 Kanchanawong Lab, MBI, NUS '+ $
    ' ' ,SPACE=3 ,XPAD=3 ,YPAD=3,modal=modal, notify_realize='initializenode')

  WID_IQUANT_DRAW = Widget_Draw(WID_BASE_node,  $
    UNAME='WID_DRAW_MAIN' ,FRAME=1 ,XOFFSET=5 ,YOFFSET=5  $
    ,SCR_XSIZE=wxsz ,SCR_YSIZE=wysz  $
    ,/BUTTON_EVENTS)


  WID_TAB_MAIN = WIDGET_TAB( WID_BASE_node, /ALIGN_right , LOCATION=0, SCR_XSIZE=570, SCR_YSIZE=1000, XOFFSET=wxsz+15, XSIZE=570, YOFFSET=10, YSIZE=1000)
  wID_BASE_tab1 = WIDGET_BASE(WID_TAB_MAIN, TITLE='Image Processing')
  wID_BASE_tab3 = WIDGET_BASE(WID_TAB_MAIN, TITLE='I/O')
  wID_BASE_tab2 = WIDGET_BASE(WID_TAB_MAIN, TITLE='Filament Tracing')
  wID_BASE_tab4 = WIDGET_BASE(WID_TAB_MAIN, TITLE='Meshwork Analysis')
  ;    wID_BASE_tab5 = WIDGET_BASE(WID_TAB_MAIN, TITLE='Parametrization')

  WID_BUTTON_EXIT = Widget_Button(wID_BASE_tab1, UNAME ='WID_BUTTON_EXIT', $
    SCr_XSIZE = 180, SCR_YSIZE = 30,/Align_center,$
    Value ='EXIT', $
    Xoffset = 10, $
    Yoffset = 940)
  WID_BUTTON_SAVESCREEN = Widget_Button(wID_BASE_tab1, UNAME ='WID_BUTTON_SAVESCREEN', $
    SCr_XSIZE = 180, SCR_YSIZE = 30,/Align_center,$
    Value ='Save Current Screen', $
    Xoffset = xoffset+10, $
    Yoffset = 940-35)

  WID_BUTTON_copyscreen = Widget_Button(wID_BASE_tab1, UNAME ='WID_BUTTON_COPYSCREEN', $
    SCr_XSIZE = 180, SCR_YSIZE = 30,/Align_center,$
    Value ='Copy Current Screen', $
    Xoffset = xoffset+10, $
    Yoffset = 940-35*2)

  WID_BUTTON_ResetZoom = Widget_Button(wID_BASE_tab1, UNAME ='WID_BUTTON_RESETZOOM', $
    SCr_XSIZE = 180, SCR_YSIZE = 30,/Align_center,$
    Value ='Unzoomed', $
    Xoffset = xoffset+10, $
    Yoffset = 5)
  WID_BUTTON_ResetZoom = Widget_Button(wID_BASE_node, UNAME ='WID_BUTTON_RESETZOOM2', $
    SCr_XSIZE = 160, SCR_YSIZE = 30,/Align_center,$
    Value ='Unzoomed', $
    Xoffset = wxsz+220, $
    Yoffset = wysz-10)
  WID_BUTTON_ZOOMOUT2X = Widget_Button(wID_BASE_node, UNAME ='WID_BUTTON_REFRESH2', $
    SCr_XSIZE = 160, SCR_YSIZE = 30,/Align_center,$
    Value ='Refresh', $
    Xoffset = wxsz+220+165, $
    Yoffset = wysz-10)

  WID_BUTTON_ZOOMOUT2X = Widget_Button(wID_BASE_tab1, UNAME ='WID_BUTTON_ZOOMOUT2X', $
    SCr_XSIZE = 180, SCR_YSIZE = 30,/Align_center,$
    Value ='Zoom Out 2X', $
    Xoffset = xoffset+10, $
    Yoffset = 5+35)
  WID_BUTTON_ZOOMOUT2X = Widget_Button(wID_BASE_tab1, UNAME ='WID_BUTTON_REFRESH', $
    SCr_XSIZE = 180, SCR_YSIZE = 30,/Align_center,$
    Value ='Refresh', $
    Xoffset = xoffset+10, $
    Yoffset = 5+35*2)
  WID_BUTTON_ZOOMOUT2X = Widget_Button(wID_BASE_tab1, UNAME ='WID_BUTTON_CLEARSCREEN', $
    SCr_XSIZE = 180, SCR_YSIZE = 30,/Align_center,$
    Value ='Clear Screen', $
    Xoffset = xoffset+10, $
    Yoffset = 5+35*3)
  ;  WID_BUTTON_ZOOMOUT2X = Widget_Button(wID_BASE_tab1, UNAME ='WID_BUTTON_SCALEBAR', $
  ;    SCr_XSIZE = 180, SCR_YSIZE = 30,/Align_center,$
  ;    Value ='Scale Bar', $
  ;    Xoffset = xoffset+10, $
  ;    Yoffset = 5+35*4)
  WID_BUTTON_ZOOMOUT2X = Widget_Button(wID_BASE_node, UNAME ='WID_BUTTON_CLEARSCREEN2', $
    SCr_XSIZE = 180, SCR_YSIZE = 30,/Align_center,$
    Value ='Clear Screen', $
    Xoffset = wxsz+220-185, $
    Yoffset = wysz-10)

  WID_BUTTON_ZOOMOUT2X = Widget_Button(wID_BASE_tab1, UNAME ='WID_BUTTON_OFTTEST', $
    SCr_XSIZE = 180, SCR_YSIZE = 30,/Align_center,$
    Value ='Test OFT', $
    Xoffset = xoffset+10, $
    Yoffset = 5+35*6+85+85)
  wid_base_imin = widget_base(wID_BASE_tab1, xoffset= xoffset+10, yoffset = 5+35*6, scr_xsize = 180, scr_ysize=80)
  wid_slider_hmin = cw_fslider(wid_base_imin, /DOUBLE,minimum=3, maximum=50.,scroll = 1, $
    title='OFT Radius (pixel)',value = 10.,xsize=250,ysize = 65,/edit, uname='WID_SLIDER_OFTRADIUS')
  wid_base_imin = widget_base(wID_BASE_tab1, xoffset= xoffset+10, yoffset = 5+35*6+85, scr_xsize = 180, scr_ysize=80)
  wid_slider_hmin = cw_fslider(wid_base_imin, /DOUBLE,minimum=15, maximum=180.,scroll = 1, $
    title='# of Sectors',value = 20.,xsize=250,ysize = 65,/edit, uname='WID_SLIDER_OFTSECTOR')

  WID_BUTTON_ZOOMOUT2X = Widget_Button(wID_BASE_tab1, UNAME ='WID_BUTTON_OFTFULL', $
    SCr_XSIZE = 180, SCR_YSIZE = 30,/Align_center,$
    Value ='Perform full OFT Calculation', $
    Xoffset = xoffset+10, $
    Yoffset = 5+35*7+85+85)


  WID_BUTTON_ZOOMOUT2X = Widget_Button(wID_BASE_tab1, UNAME ='WID_BUTTON_OFTHISTOGRAM', $
    SCr_XSIZE = 180, SCR_YSIZE = 30,/Align_center,$
    Value ='Histogram of OFT', $
    Xoffset = xoffset+10+180+5+180+5, $
    Yoffset = 5+35*6)

  WID_BUTTON_ZOOMOUT2X = Widget_Button(wID_BASE_tab1, UNAME ='WID_BUTTON_OFTOTSU', $
    SCr_XSIZE = 180, SCR_YSIZE = 30,/Align_center,$
    Value ='Calculate Threshold', $
    Xoffset = xoffset+10+180+5+180+5, $
    Yoffset = 5+35*7)
  wid_base_imin = widget_base(wID_BASE_tab1, xoffset= xoffset+10+180+5+180+5, yoffset = 5+35*8, scr_xsize = 180, scr_ysize=80)
  wid_slider_hmin = cw_fslider(wid_base_imin, /DOUBLE,minimum=0.1, maximum=50000.,scroll = 0.1, $
    title='OFT Threshold Level',value = 100.,xsize=250,ysize = 65,/edit, uname='WID_SLIDER_OFTTHRESHOLD')
  wid_base_imin = widget_base(wID_BASE_tab1, xoffset= xoffset+10+180+5+180+5, yoffset = 5+35*8+85, scr_xsize = 180, scr_ysize=80)
  wid_slider_hmin = cw_fslider(wid_base_imin, /DOUBLE,minimum=0.005, maximum=20.,scroll = 0.1, $
    title='OFT Threshold Multiplier',value = 0.4,xsize=250,ysize = 65,/edit, uname='WID_SLIDER_OFTTHRESHOLDMULTIPLIER')

  WID_BUTTON_ZOOMOUT2X = Widget_Button(wID_BASE_tab1, UNAME ='WID_BUTTON_OVERLAYOFTTHRESHOLD', $
    SCr_XSIZE = 180, SCR_YSIZE = 30,/Align_center,$
    Value ='OFT/Threshold overlay', $
    Xoffset = xoffset+10+180+5+180+5, $
    Yoffset = 5+35*8+85+85)
  ;  WID_BUTTON_ZOOMOUT2X = Widget_Button(wID_BASE_tab1, UNAME ='WID_BUTTON_SKELETONSMLM', $
  ;    SCr_XSIZE = 180, SCR_YSIZE = 30,/Align_center,$
  ;    Value ='Skeleton/SMLM Overlay', $
  ;    Xoffset = xoffset+10+180+5+180+5, $
  ;    Yoffset = 5+35*9+85+85)
  WID_BUTTON_ZOOMOUT2X = Widget_Button(wID_BASE_tab1, UNAME ='WID_BUTTON_HMINIMAOFT', $
    SCr_XSIZE = 180, SCR_YSIZE = 30,/Align_center,$
    Value ='H-minima/OFT Overlay', $
    Xoffset = xoffset+10+180+5+180+5, $
    Yoffset = 5+35*9+85+85)
  WID_BUTTON_ZOOMOUT2X = Widget_Button(wID_BASE_tab1, UNAME ='WID_BUTTON_HMINIMAOFTWATERSHED', $
    SCr_XSIZE = 180, SCR_YSIZE = 30,/Align_center,$
    Value ='Segmented Filaments Overlay', $
    Xoffset = xoffset+10+180+5+180+5, $
    Yoffset = 5+35*10+85+85)
  WID_BUTTON_ZOOMOUT2X = Widget_Button(wID_BASE_tab1, UNAME ='WID_BUTTON_EXTRACTFILAMENT', $
    SCr_XSIZE = 180, SCR_YSIZE = 30,/Align_center,$
    Value ='Perform Pore Extraction', $
    Xoffset = xoffset+10+180+5+180+5, $
    Yoffset = 5+35*11+85+85)


  WID_BUTTON_ZOOMOUT2X = Widget_Button(wID_BASE_tab1, UNAME ='WID_BUTTON_SEGMENTATION', $
    SCr_XSIZE = 180, SCR_YSIZE = 30,/Align_center,$
    Value ='Define Analysis ROI', $
    Xoffset = xoffset+10, $
    Yoffset = 940-35*7)
  WID_BUTTON_ZOOMOUT2X = Widget_Button(wID_BASE_tab1, UNAME ='WID_BUTTON_EDITSEGMENTATION', $
    SCr_XSIZE = 180, SCR_YSIZE = 30,/Align_center,$
    Value ='Edit Analysis ROI', $
    Xoffset = xoffset+10, $
    Yoffset = 940-35*6)
  WID_BUTTON_ZOOMOUT2X = Widget_Button(wID_BASE_tab1, UNAME ='WID_BUTTON_RESETSEGMENTATION', $
    SCr_XSIZE = 180, SCR_YSIZE = 30,/Align_center,$
    Value ='Reset ROI', $
    Xoffset = xoffset+10, $
    Yoffset = 940-35*5)

  WID_BUTTON_ZOOMOUT2X = Widget_Button(wID_BASE_tab1, UNAME ='WID_BUTTON_EDITWATERSHED', $
    SCr_XSIZE = 180, SCR_YSIZE = 30,/Align_center,$
    Value ='Edit Watershed Image', $
    Xoffset = xoffset+10+180+5+180+5, $
    Yoffset = 940-35*7)
  WID_BUTTON_ZOOMOUT2X = Widget_Button(wID_BASE_tab1, UNAME ='WID_BUTTON_SEGMENTATIONASTERS', $
    SCr_XSIZE = 180, SCR_YSIZE = 30,/Align_center,$
    Value ='Define Asters ROI', $
    Xoffset = xoffset+10+185, $
    Yoffset = 940-35*7)
  WID_BUTTON_ZOOMOUT2X = Widget_Button(wID_BASE_tab1, UNAME ='WID_BUTTON_EDITSEGMENTATIONASTERS', $
    SCr_XSIZE = 180, SCR_YSIZE = 30,/Align_center,$
    Value ='Edit Asters ROI', $
    Xoffset = xoffset+10+185, $
    Yoffset = 940-35*6)



  WID_DROPLIST_color = Widget_Droplist(wID_BASE_tab1,  $
    UNAME='WID_DROPLIST_COLOR' ,XOFFSET=xoffset+10+180+5+180+5,YOFFSET=5 ,SCR_XSIZE=180  $
    ,SCR_YSIZE=25 ,TITLE='Color :' ,VALUE=colordroplist)

  WID_DROPLIST_display = Widget_Droplist(wID_BASE_tab1,  $
    UNAME='WID_DROPLIST_DISPLAY' ,XOFFSET=xoffset+10+180+5+180+5,YOFFSET=5+35 ,SCR_XSIZE=180  $
    ,SCR_YSIZE=25 ,TITLE='Display :' ,VALUE=displaylist)
  WID_BUTTON_ZOOMOUT2X = Widget_Button(wID_BASE_tab1, UNAME ='WID_BUTTON_QUARTERVIEW', $
    SCr_XSIZE = 180, SCR_YSIZE = 30,/Align_center,$
    Value ='Quater-View I', $
    Xoffset = xoffset+10+180+5+180+5, $
    Yoffset = 5+35*2)
  WID_BUTTON_ZOOMOUT2X = Widget_Button(wID_BASE_tab1, UNAME ='WID_BUTTON_QUARTERVIEW2', $
    SCr_XSIZE = 180, SCR_YSIZE = 30,/Align_center,$
    Value ='Quater-View II', $
    Xoffset = xoffset+10+180+5+180+5, $
    Yoffset = 5+35*3)


  wid_base_imin = widget_base(wID_BASE_tab1, xoffset= xoffset+10+180+5+180+5, yoffset = 900, scr_xsize = 180, scr_ysize=80)
  wid_slider_hmin = cw_fslider(wid_base_imin, /DOUBLE,minimum=0.02, maximum=200.,scroll = 0.1, $
    title='SMLM Pixel Size (nm)',value = 20,xsize=250,ysize = 65,/edit, uname='WID_SLIDER_SMLMPIXELSIZE')


  ;=========================I/O==============

  WID_TEXT_cust = Widget_Text(WID_BASE_tab3,  $
    UNAME='WID_TEXT_IO' ,FRAME=1 ,XOFFSET=5 ,YOFFSET=5  $
    ,SCR_XSIZE=540 ,SCR_YSIZE=80 ,/WRAP ,VALUE=[''] ,XSIZE=200 ,YSIZE=2,editable=0)

  WID_TEXT_cust = Widget_Text(WID_BASE_tab3,  $
    UNAME='WID_TEXT_MATLAB' ,FRAME=1 ,XOFFSET=5 ,YOFFSET=905  $
    ,SCR_XSIZE=540 ,SCR_YSIZE=40 ,/WRAP ,VALUE=['Matlab_Application_23.2'] ,XSIZE=200 ,YSIZE=2,editable=1)
  WID_DROPLIST_color = Widget_Droplist(wID_BASE_tab3,  $
    UNAME='WID_DROPLIST_MATLAB' ,XOFFSET=xoffset+10+180+5+180+5,YOFFSET=905-35 ,SCR_XSIZE=180  $
    ,SCR_YSIZE=25 ,TITLE='Matlab Registry :' ,VALUE=['Default','Specified'])
  WID_BUTTON_ResetZoom = Widget_Button(wID_BASE_tab3, UNAME ='WID_BUTTON_LOADTIFF', $
    SCr_XSIZE = 180, SCR_YSIZE = 30,/Align_center,$
    Value ='Load SMLM image (*.tiff)', $
    Xoffset = xoffset+10, $
    Yoffset = 5+100)

  WID_BUTTON_ResetZoom = Widget_Button(wID_BASE_tab3, UNAME ='WID_BUTTON_LOADOFT', $
    SCr_XSIZE = 190, SCR_YSIZE = 30,/Align_center,$
    Value ='Load OFT image (.csv export)', $
    Xoffset = xoffset+10+180+5, $
    Yoffset = 5+35*0+100)
  WID_BUTTON_ResetZoom = Widget_Button(wID_BASE_tab3, UNAME ='WID_BUTTON_LOADFILAMENTMATLAB', $
    SCr_XSIZE = 190, SCR_YSIZE = 30,/Align_center,$
    Value ='Load FilamentTrace (.csv export)', $
    Xoffset = xoffset+10+180+5, $
    Yoffset = 5+35*1+100)
  WID_BUTTON_ResetZoom = Widget_Button(wID_BASE_tab3, UNAME ='WID_BUTTON_EXPORTCROP', $
    SCr_XSIZE = 190, SCR_YSIZE = 30,/Align_center,$
    Value ='Export ZoomedArea (meshworks.sav)', $
    Xoffset = xoffset+10+180+5, $
    Yoffset = 5+35*2+100)

  WID_BUTTON_ResetZoom = Widget_Button(wID_BASE_tab3, UNAME ='WID_BUTTON_SAVEOFT', $
    SCr_XSIZE = 170, SCR_YSIZE = 30,/Align_center,$
    Value ='Export OFT image', $
    Xoffset = xoffset+10*2+2*180+5*2, $
    Yoffset = 5+35*0+100)
  WID_BUTTON_ResetZoom = Widget_Button(wID_BASE_tab3, UNAME ='WID_BUTTON_SAVELFT', $
    SCr_XSIZE = 170, SCR_YSIZE = 30,/Align_center,$
    Value ='Export LFT image', $
    Xoffset = xoffset+10*2+2*180+5*2, $
    Yoffset = 5+35*1+100)
  WID_BUTTON_ResetZoom = Widget_Button(wID_BASE_tab3, UNAME ='WID_BUTTON_SAVEORIENTATION', $
    SCr_XSIZE = 170, SCR_YSIZE = 30,/Align_center,$
    Value ='Export Orientation image', $
    Xoffset = xoffset+10*2+2*180+5*2, $
    Yoffset = 5+35*2+100)


  WID_BUTTON_ResetZoom = Widget_Button(wID_BASE_tab3, UNAME ='WID_BUTTON_SAVESAV', $
    SCr_XSIZE = 180, SCR_YSIZE = 30,/Align_center,$
    Value ='Save analysis to *meshworks.sav', $
    Xoffset = xoffset+10, $
    Yoffset = 5+35*2+100)

  WID_BUTTON_ResetZoom = Widget_Button(wID_BASE_tab3, UNAME ='WID_BUTTON_LOADSAV', $
    SCr_XSIZE = 180, SCR_YSIZE = 30,/Align_center,$
    Value ='Load analysis from *meshworks.sav', $
    Xoffset = xoffset+10, $
    Yoffset = 5+35*1+100)

  ;======

  WID_BUTTON_ResetZoom = Widget_Button(wID_BASE_tab2, UNAME ='WID_BUTTON_UPDATETRACING', $
    SCr_XSIZE = 180, SCR_YSIZE = 30,/Align_center,$
    Value ='Update Filaments and Pores Tables', $
    Xoffset = xoffset+10, $
    Yoffset = yoffset+5)
  WID_BUTTON_ResetZoom = Widget_Button(wID_BASE_tab2, UNAME ='WID_BUTTON_UPDATEALL', $
    SCr_XSIZE = 180, SCR_YSIZE = 30,/Align_center,$
    Value ='Update All', $
    Xoffset = xoffset+10+185, $
    Yoffset = yoffset+5)
  WID_BUTTON_EXIT = Widget_Button(wID_BASE_tab2, UNAME ='WID_BUTTON_UPDATEALLBATCH', $
    SCr_XSIZE = 180, SCR_YSIZE = 30,/Align_center,$
    Value ='Update All/Batch Mode', $
    Xoffset = 10, $
    Yoffset = 940)

  WID_Table_pores= Widget_Table(WID_BASE_tab2,  $
    UNAME='WID_TABLE_NODES' ,xoffset = 25, YOFFSET=5+35 ,SCR_XSIZE=500  $
    ,SCR_YSIZE=150 ,COLUMN_LABELS=[ 'Node-ID', 'X-center','Y-center', 'SMLM intensity','N-code','F-ID1','F-ID2','F-ID3','F-ID4','F-ID5','F-ID6','F-ID7','F-ID8', $
    'P-ID1','P-ID2','P-ID3','P-ID4','P-ID5','P-ID6','P-ID7','P-ID8'] ,$
    XSIZE=21 ,YSIZE=1000,column_widths=replicate(60, 21),/disjoint_selection,value=tablearray,format='(F12.3)')


  WID_Table_pores= Widget_Table(WID_BASE_tab2,  $
    UNAME='WID_TABLE_FILAMENTS' ,xoffset = 25, YOFFSET=5+300+35 ,SCR_XSIZE=500  $
    ,SCR_YSIZE=150 ,COLUMN_LABELS=[ 'filament-ID', 'Node-I','Node-F','F-code','X-i','Y-i','X-f','Y-f','Contour-length','Angle-I','Angle-F','GrAng-I','GrAng-F','Avg-SMLM-intensity'] ,$
    XSIZE=14 ,YSIZE=1000,column_widths=replicate(60, 14),/disjoint_selection,value=tablearray,format='(F12.3)')


  WID_Table_pores= Widget_Table(WID_BASE_tab2,  $
    UNAME='WID_TABLE_PORES' ,xoffset = 25, YOFFSET=5+35+600 ,SCR_XSIZE=500  $
    ,SCR_YSIZE=150 ,COLUMN_LABELS=[ 'pore-ID', 'X-center','Y-center','Area','Vertices','P-Code','P-avg-smlm','P-max-smlm','P-edge-avg'] ,$
    XSIZE=9 ,YSIZE=1000,column_widths=replicate(60, 9),/disjoint_selection,value=tablearray,format='(F12.3)')

  WID_BUTTON_Pinpoint = Widget_Button(wID_BASE_tab2, UNAME ='WID_BUTTON_PLOTNODES', $
    SCr_XSIZE = 100, SCR_YSIZE = 30,/Align_center,$
    Value ='Plot Nodes', $
    Xoffset = xoffset+10, $
    Yoffset = 5+35+150+5)
  WID_BUTTON_Pinpoint = Widget_Button(wID_BASE_tab2, UNAME ='WID_BUTTON_PLOTNODESID', $
    SCr_XSIZE = 100, SCR_YSIZE = 30,/Align_center,$
    Value ='Show Node-ID', $
    Xoffset = xoffset+10+105, $
    Yoffset = 5+35+150+5)
  WID_BUTTON_Pinpoint = Widget_Button(wID_BASE_tab2, UNAME ='WID_BUTTON_PINPOINTNODE', $
    SCr_XSIZE = 100, SCR_YSIZE = 30,/Align_center,$
    Value ='Show Selected', $
    Xoffset = xoffset+10+105*2, $
    Yoffset = 5+35+150+5)
  WID_BUTTON_Pinpoint = Widget_Button(wID_BASE_tab2, UNAME ='WID_BUTTON_SORTNODE', $
    SCr_XSIZE = 100, SCR_YSIZE = 30,/Align_center,$
    Value ='Sort by Col. >>>', $
    Xoffset = xoffset+10+105*3, $
    Yoffset = 5+35+150+5)
  WID_BUTTON_Pinpoint = Widget_Button(wID_BASE_tab2, UNAME ='WID_BUTTON_SORTNODEASCEND', $
    SCr_XSIZE = 100, SCR_YSIZE = 30,/Align_center,$
    Value ='Sort by Col. <<<', $
    Xoffset = xoffset+10+105*3, $
    Yoffset = 5+35+150+5+35)
  WID_BUTTON_Pinpoint = Widget_Button(wID_BASE_tab2, UNAME ='WID_BUTTON_EXPORTNODE', $
    SCr_XSIZE = 100, SCR_YSIZE = 30,/Align_center,$
    Value ='Export Table (.csv)', $
    Xoffset = xoffset+10+105*4, $
    Yoffset = 5+35+150+5)
  WID_BUTTON_Pinpoint = Widget_Button(wID_BASE_tab2, UNAME ='WID_BUTTON_GOTONODE', $
    SCr_XSIZE = 100, SCR_YSIZE = 30,/Align_center,$
    Value ='Go to Selected', $
    Xoffset = xoffset+10+105*4, $
    Yoffset = 5+35+150+5+35)
  WID_BUTTON_Pinpoint = Widget_Button(wID_BASE_tab2, UNAME ='WID_BUTTON_STATNODES', $
    SCr_XSIZE = 100, SCR_YSIZE = 30,/Align_center,$
    Value ='Node Statistics', $
    Xoffset = xoffset+10, $
    Yoffset = 5+35+150+5+35)
  WID_BUTTON_Pinpoint = Widget_Button(wID_BASE_tab2, UNAME ='WID_BUTTON_FITNODES', $
    SCr_XSIZE = 205, SCR_YSIZE = 30,/Align_center,$
    Value ='Analyze Selected Nodes', $
    Xoffset = xoffset+10+105, $
    Yoffset = 5+35+150+5+35)

  WID_BUTTON_Pinpoint = Widget_Button(wID_BASE_tab2, UNAME ='WID_BUTTON_PLOTFILAMENTS', $
    SCr_XSIZE = 100, SCR_YSIZE = 30,/Align_center,$
    Value ='Plot Filaments', $
    Xoffset = xoffset+10, $
    Yoffset = 5+35+150+300+5)
  WID_BUTTON_Pinpoint = Widget_Button(wID_BASE_tab2, UNAME ='WID_BUTTON_PLOTFILAMENTSID', $
    SCr_XSIZE = 100, SCR_YSIZE = 30,/Align_center,$
    Value ='Show Filament-ID', $
    Xoffset = xoffset+10+105, $
    Yoffset = 5+35+150+300+5)
  WID_BUTTON_Pinpoint = Widget_Button(wID_BASE_tab2, UNAME ='WID_BUTTON_PINPOINTFILAMENTS', $
    SCr_XSIZE = 100, SCR_YSIZE = 30,/Align_center,$
    Value ='Show Selected', $
    Xoffset = xoffset+10+105*2, $
    Yoffset = 5+35+150+300+5)
  WID_BUTTON_Pinpoint = Widget_Button(wID_BASE_tab2, UNAME ='WID_BUTTON_SORTFILAMENTS', $
    SCr_XSIZE = 100, SCR_YSIZE = 30,/Align_center,$
    Value ='Sort by Col. >>>', $
    Xoffset = xoffset+10+105*3, $
    Yoffset = 5+35+150+300+5)
  WID_BUTTON_Pinpoint = Widget_Button(wID_BASE_tab2, UNAME ='WID_BUTTON_SORTFILAMENTSASCEND', $
    SCr_XSIZE = 100, SCR_YSIZE = 30,/Align_center,$
    Value ='Sort by Col. <<<', $
    Xoffset = xoffset+10+105*3, $
    Yoffset = 5+35+150+300+5+35)
  WID_BUTTON_Pinpoint = Widget_Button(wID_BASE_tab2, UNAME ='WID_BUTTON_EXPORTFILAMENTS', $
    SCr_XSIZE = 100, SCR_YSIZE = 30,/Align_center,$
    Value ='Export Table (.csv)', $
    Xoffset = xoffset+10+105*4, $
    Yoffset = 5+35+150+300+5)
  WID_BUTTON_Pinpoint = Widget_Button(wID_BASE_tab2, UNAME ='WID_BUTTON_GOTOFILAMENT', $
    SCr_XSIZE = 100, SCR_YSIZE = 30,/Align_center,$
    Value ='Go to selected', $
    Xoffset = xoffset+10+105*4, $
    Yoffset = 5+35+150+300+5+35)
  WID_BUTTON_Pinpoint = Widget_Button(wID_BASE_tab2, UNAME ='WID_BUTTON_STATFILAMENTS', $
    SCr_XSIZE = 100, SCR_YSIZE = 30,/Align_center,$
    Value ='Flmnt Statistics', $
    Xoffset = xoffset+10, $
    Yoffset = 5+35+150+300+5+35)
  WID_BUTTON_Pinpoint = Widget_Button(wID_BASE_tab2, UNAME ='WID_BUTTON_FITFILAMENTS', $
    SCr_XSIZE = 205, SCR_YSIZE = 30,/Align_center,$
    Value ='Analyze Selected Filaments', $
    Xoffset = xoffset+10+105, $
    Yoffset = 5+35+150+300+5+35)


  WID_BUTTON_Pinpoint = Widget_Button(wID_BASE_tab2, UNAME ='WID_BUTTON_PLOTPORES', $
    SCr_XSIZE = 100, SCR_YSIZE = 30,/Align_center,$
    Value ='Plot Pores', $
    Xoffset = xoffset+10, $
    Yoffset = 5+35+150+300*2+5)
  WID_BUTTON_Pinpoint = Widget_Button(wID_BASE_tab2, UNAME ='WID_BUTTON_PLOTPORESID', $
    SCr_XSIZE = 100, SCR_YSIZE = 30,/Align_center,$
    Value ='Show Pore-ID', $
    Xoffset = xoffset+10+105, $
    Yoffset = 5+35+150+300*2+5)
  WID_BUTTON_Pinpoint = Widget_Button(wID_BASE_tab2, UNAME ='WID_BUTTON_PINPOINTPORES', $
    SCr_XSIZE = 100, SCR_YSIZE = 30,/Align_center,$
    Value ='Show Selected', $
    Xoffset = xoffset+10+105*2, $
    Yoffset = 5+35+150+300*2+5)
  WID_BUTTON_Pinpoint = Widget_Button(wID_BASE_tab2, UNAME ='WID_BUTTON_SORTPORES', $
    SCr_XSIZE = 100, SCR_YSIZE = 30,/Align_center,$
    Value ='Sort by Col. >>>', $
    Xoffset = xoffset+10+105*3, $
    Yoffset = 5+35+150+300*2+5)
  WID_BUTTON_Pinpoint = Widget_Button(wID_BASE_tab2, UNAME ='WID_BUTTON_SORTPORESASCEND', $
    SCr_XSIZE = 100, SCR_YSIZE = 30,/Align_center,$
    Value ='Sort by Col. <<<', $
    Xoffset = xoffset+10+105*3, $
    Yoffset = 5+35+150+300*2+5+35)
  WID_BUTTON_Pinpoint = Widget_Button(wID_BASE_tab2, UNAME ='WID_BUTTON_EXPORTPORES', $
    SCr_XSIZE = 100, SCR_YSIZE = 30,/Align_center,$
    Value ='Export Table (.csv)', $
    Xoffset = xoffset+10+105*4, $
    Yoffset = 5+35+150+300*2+5)
  WID_BUTTON_Pinpoint = Widget_Button(wID_BASE_tab2, UNAME ='WID_BUTTON_GOTOPORE', $
    SCr_XSIZE = 100, SCR_YSIZE = 30,/Align_center,$
    Value ='Go to selected', $
    Xoffset = xoffset+10+105*4, $
    Yoffset = 5+35+150+300*2+5+35)
  WID_BUTTON_Pinpoint = Widget_Button(wID_BASE_tab2, UNAME ='WID_BUTTON_STATPORES', $
    SCr_XSIZE = 100, SCR_YSIZE = 30,/Align_center,$
    Value ='Pore Statistics', $
    Xoffset = xoffset+10, $
    Yoffset = 5+35+150+300*2+5+35)
  WID_BUTTON_Pinpoint = Widget_Button(wID_BASE_tab2, UNAME ='WID_BUTTON_TOGGLEPORES', $
    SCr_XSIZE = 205, SCR_YSIZE = 30,/Align_center,$
    Value ='Toggle Pore-Code', $
    Xoffset = xoffset+10+105, $
    Yoffset = 5+35+150+300*2+5+35)


  ;================

  WID_BUTTON_ResetZoom = Widget_Button(wID_BASE_tab4, UNAME ='WID_BUTTON_UPDATEANGLES', $
    SCr_XSIZE = 180, SCR_YSIZE = 30,/Align_center,$
    Value ='Update Angles Tables', $
    Xoffset = xoffset+10, $
    Yoffset = yoffset+5)

  WID_Table_pores= Widget_Table(WID_BASE_tab4,  $
    UNAME='WID_TABLE_ANGLES' ,xoffset = 25, YOFFSET=5+35 ,SCR_XSIZE=500  $
    ,SCR_YSIZE=150 ,COLUMN_LABELS=[ 'Angle-ID', 'Node-ID','X-center','Y-center', 'Fit-If-Angle','Direct-If-Angle','Opt-If-Angle','Fil-ID1','Fil-ID2','Fil-1-length','Fil-2-length','Angle-Code'] ,$
    XSIZE=12 ,YSIZE=1000,column_widths=replicate(60, 21),/disjoint_selection,value=tablearray,format='(F12.3)')
  WID_BUTTON_Pinpoint = Widget_Button(wID_BASE_tab4, UNAME ='WID_BUTTON_GOTOANGLES', $
    SCr_XSIZE = 205, SCR_YSIZE = 30,/Align_center,$
    Value ='Go to Angles', $
    Xoffset = xoffset+10, $
    Yoffset = 5+35+150+5)
  WID_BUTTON_Pinpoint = Widget_Button(wID_BASE_tab4, UNAME ='WID_BUTTON_PINPOINTANGLES', $
    SCr_XSIZE = 100, SCR_YSIZE = 30,/Align_center,$
    Value ='Show Selected', $
    Xoffset = xoffset+10+105*2, $
    Yoffset = 5+35+150+5)
  WID_BUTTON_Pinpoint = Widget_Button(wID_BASE_tab4, UNAME ='WID_BUTTON_SORTANGLES', $
    SCr_XSIZE = 100, SCR_YSIZE = 30,/Align_center,$
    Value ='Sort by Col. >>>', $
    Xoffset = xoffset+10+105*3, $
    Yoffset = 5+35+150+5)
  WID_BUTTON_Pinpoint = Widget_Button(wID_BASE_tab4, UNAME ='WID_BUTTON_SORTANGLESASCEND', $
    SCr_XSIZE = 100, SCR_YSIZE = 30,/Align_center,$
    Value ='Sort by Col. <<<', $
    Xoffset = xoffset+10+105*3, $
    Yoffset = 5+35+150+5+35)
  WID_BUTTON_Pinpoint = Widget_Button(wID_BASE_tab4, UNAME ='WID_BUTTON_EXPORTANGLES', $
    SCr_XSIZE = 100, SCR_YSIZE = 30,/Align_center,$
    Value ='Export Table (.csv)', $
    Xoffset = xoffset+10+105*4, $
    Yoffset = 5+35+150+5)
  WID_BUTTON_Pinpoint = Widget_Button(wID_BASE_tab4, UNAME ='WID_BUTTON_STATANGLES', $
    SCr_XSIZE = 205, SCR_YSIZE = 30,/Align_center,$
    Value ='Angle Statistics', $
    Xoffset = xoffset+10, $
    Yoffset = 5+35+150+5+35)
  WID_BUTTON_Pinpoint = Widget_Button(wID_BASE_tab4, UNAME ='WID_BUTTON_SHOWANGLES', $
    SCr_XSIZE = 100, SCR_YSIZE = 30,/Align_center,$
    Value ='Show Angles', $
    Xoffset = xoffset+10+105*2, $
    Yoffset = 5+35*2+150+5)



  WID_Table_pores= Widget_Table(WID_BASE_tab4,  $
    UNAME='WID_TABLE_ASTERS' ,xoffset = 25, YOFFSET=5+35+300 ,SCR_XSIZE=500  $
    ,SCR_YSIZE=150 ,COLUMN_LABELS=[ 'Aster-ID', 'X-center','Y-center', 'Area','AvgIntens','PeakIntens','Coord-#','aster-code'] ,$
    XSIZE=8 ,YSIZE=1000,column_widths=replicate(63, 8),/disjoint_selection,value=tablearray,format='(F12.3)',/editable)
  WID_BUTTON_Pinpoint = Widget_Button(wID_BASE_tab4, UNAME ='WID_BUTTON_GOTOASTERS', $
    SCr_XSIZE = 205, SCR_YSIZE = 30,/Align_center,$
    Value ='Go to Asters', $
    Xoffset = xoffset+10, $
    Yoffset = 5+35+150+5+300)
  WID_BUTTON_ResetZoom = Widget_Button(wID_BASE_tab4, UNAME ='WID_BUTTON_UPDATEASTERS', $
    SCr_XSIZE = 180, SCR_YSIZE = 30,/Align_center,$
    Value ='Update Asters Tables', $
    Xoffset = xoffset+10, $
    Yoffset = yoffset+5+300)
  WID_BUTTON_Pinpoint = Widget_Button(wID_BASE_tab4, UNAME ='WID_BUTTON_PINPOINTASTERS', $
    SCr_XSIZE = 100, SCR_YSIZE = 30,/Align_center,$
    Value ='Show Selected', $
    Xoffset = xoffset+10+105*2, $
    Yoffset = 5+35+150+5+300)
  WID_BUTTON_Pinpoint = Widget_Button(wID_BASE_tab4, UNAME ='WID_BUTTON_SORTASTERS', $
    SCr_XSIZE = 100, SCR_YSIZE = 30,/Align_center,$
    Value ='Sort by Col. >>>', $
    Xoffset = xoffset+10+105*3, $
    Yoffset = 5+35+150+5+300)
  WID_BUTTON_Pinpoint = Widget_Button(wID_BASE_tab4, UNAME ='WID_BUTTON_SORTASTERSASCEND', $
    SCr_XSIZE = 100, SCR_YSIZE = 30,/Align_center,$
    Value ='Sort by Col. <<<', $
    Xoffset = xoffset+10+105*3, $
    Yoffset = 5+35+150+5+35+300)
  WID_BUTTON_Pinpoint = Widget_Button(wID_BASE_tab4, UNAME ='WID_BUTTON_EXPORTASTERS', $
    SCr_XSIZE = 100, SCR_YSIZE = 30,/Align_center,$
    Value ='Export Table (.csv)', $
    Xoffset = xoffset+10+105*4, $
    Yoffset = 5+35+150+5+300)
  ;    WID_BUTTON_Pinpoint = Widget_Button(wID_BASE_tab4, UNAME ='WID_BUTTON_ASTERSAZIMUTH', $
  ;      SCr_XSIZE = 100, SCR_YSIZE = 30,/Align_center,$
  ;      Value ='Azimuth Corr.', $
  ;      Xoffset = xoffset+10+105*4, $
  ;      Yoffset = 5+35*2+150+5+300)
  WID_BUTTON_Pinpoint = Widget_Button(wID_BASE_tab4, UNAME ='WID_BUTTON_STATASTERS', $
    SCr_XSIZE = 205, SCR_YSIZE = 30,/Align_center,$
    Value ='Asters Statistics', $
    Xoffset = xoffset+10, $
    Yoffset = 5+35+150+5+35+300)
  WID_BUTTON_Pinpoint = Widget_Button(wID_BASE_tab4, UNAME ='WID_BUTTON_MONTAGEASTERS', $
    SCr_XSIZE = 100, SCR_YSIZE = 30,/Align_center,$
    Value ='Montage View', $
    Xoffset = xoffset+10+105*2, $
    Yoffset = 5+35*2+150+5+300)


  WID_BUTTON_Pinpoint = Widget_Button(wID_BASE_tab4, UNAME ='WID_BUTTON_EXPORTCELLHISTO', $
    SCr_XSIZE = 310, SCR_YSIZE = 30,/Align_center,$
    Value ='Cell: Export Histo. Pore/Fil-Seg-Length/Angle (.csv)', $
    Xoffset = 10, $
    Yoffset = 900-35)
  WID_BUTTON_Pinpoint = Widget_Button(wID_BASE_tab4, UNAME ='WID_BUTTON_EXPORTPOOLEDHISTO', $
    SCr_XSIZE = 310, SCR_YSIZE = 30,/Align_center,$
    Value ='Batch/Export Pooled Histo. Pore/Fil-Seg-Length/Angle (.csv)', $
    Xoffset = 10, $
    Yoffset = 900+35*0)
  WID_BUTTON_Pinpoint = Widget_Button(wID_BASE_tab4, UNAME ='WID_BUTTON_EXPORTPOOLEDDATAPTS', $
    SCr_XSIZE = 310, SCR_YSIZE = 30,/Align_center,$
    Value ='Batch/Export Pooled DataPts. Pore/Fil-Seg-Length/Angle (.csv)', $
    Xoffset = 10, $
    Yoffset = 900+35*1)

  ;    WID_BUTTON_Pinpoint = Widget_Button(wID_BASE_tab4, UNAME ='WID_BUTTON_CHISQTEST', $
  ;      SCr_XSIZE = 205, SCR_YSIZE = 30,/Align_center,$
  ;      Value ='Chi-Sq Test: nonhdr-Histo (.csv)', $
  ;      Xoffset = xoffset+10+105*3, $
  ;      Yoffset = 500)

  WID_DROPLIST_color = Widget_Droplist(wID_BASE_tab4,  $
    UNAME='WID_DROPLIST_HEADER' ,XOFFSET=10,YOFFSET=900-35*2 ,SCR_XSIZE=310  $
    ,SCR_YSIZE=25 ,TITLE='File-Header:' ,VALUE=['Header','No-header']);,'Add Aster'])

  WID_BUTTON_EXIT = Widget_Button(wID_BASE_tab4, UNAME ='WID_BUTTON_SUMMARY', $
    SCr_XSIZE = 310, SCR_YSIZE = 30,/Align_center,$
    Value ='Summary: Meshwork Properties', $
    Xoffset = 10, $
    Yoffset = 700-35)
  WID_BUTTON_EXIT = Widget_Button(wID_BASE_tab4, UNAME ='WID_BUTTON_SUMMARYBATCH', $
    SCr_XSIZE = 310, SCR_YSIZE = 30,/Align_center,$
    Value ='Batch Mode/Summary: Meshwork Properties', $
    Xoffset = 10, $
    Yoffset = 700)



  Widget_Control, /REALIZE, WID_BASE_node
  XManager, 'WID_BASE_NODE', WID_BASE_node,/NO_BLOCK

end

pro meshwork_drawevents, event,frame=frame
  common display_info, def_w, wxsz,wysz, mousedown, infoinitial, frontendwindow, frontendzoom
  Common meshwork_dataset, properties, image_set, filamentmask, nodemask, node_table, pore_table, filament_table, angle_table, aster_table

  displaymode = widget_info(Widget_Info(event.top, FIND_BY_UNAME='WID_DROPLIST_DISPLAY'),/droplist_select)
  mousemode = 0; widget_info(Widget_Info(event.top, FIND_BY_UNAME='WID_DROPLIST_MOUSE'),/droplist_select)
  if ~keyword_set(frame) then frame = 0
  width=properties.xpixels
  height=properties.ypixels
  mgw = wxsz/width
  drawwindow = !d.window

  IF event.type GT 2 THEN RETURN
  eventTypes = ['DOWN', 'UP', 'MOTION']
  thisEvent = eventTypes[event.type]
  Widget_Control, event.top, Get_UValue=info, /No_Copy

  z = meshwork_getzoomfactor()


  if size(info,/type) ne 8 then begin
    info = infoinitial
    widget_control, event.top, set_uvalue=info,/no_copy
    print, 'Reinitialized info..'
    return
  end

  case thisevent of
    'DOWN':begin
      widget_control, info.drawID, draw_motion_events =1
      window, /free,/pixmap,xsize=info.xsize,ysize=info.ysize
      info.pixID = !D.window
      device,copy=[0,0,info.xsize,info.ysize,0,0,info.wid]
      info.sx = event.x >0
      info.sy = event.y >0
      wset, def_w
      mousedown = 1
      widget_control, event.top, set_uvalue=info,/no_copy
      ;print, 'down'
    end
    'MOTION': begin
      wset,info.wid
      device,copy=[0,0,info.xsize,info.ysize,0,0,info.pixID]
      sx=info.sx
      sy=info.sy
      dx=(event.x < wxsz)>0
      dy= (event.y < wysz)>0
      ; forcing square
      dim = abs(dx-sx)<abs(dy-sy)
      dx = (event.x gt sx) ? info.sx+dim:info.sx-dim
      dy = (event.y gt sy) ? info.sy+dim:info.sy-dim
      plotS, [sx,sx,dx,dx,sx],[sy,dy,dy,sy,sy],/device, color =info.boxcolor
      wset, def_w
      widget_control, event.top, set_uvalue=info,/no_copy
    end
    'UP': begin
      if mousedown eq 0 then begin
        widget_control, event.top, set_uvalue=info,/no_copy
        return
      endif
      ;      if size(info,/type) ne 8 then return
      mousedown = 0
      if info.pixID eq -1 then begin
        window, /free,/pixmap,xsize=info.xsize,ysize=info.ysize
        info.pixID = !D.window
        widget_control, event.top, set_uvalue=info,/no_copy
        return
      end
      device, window_state= windowstate
      if windowstate[info.pixID] lt 1 then return
      wset, info.wid

      device,copy=[0,0,info.xsize,info.ysize,0,0,info.pixID]
      wdelete, info.pixID
      widget_control, info.drawID, draw_motion_events =0, clear_events=1
      dx=(event.x < wxsz)>0
      dy= (event.y < wysz)>0
      sx=info.sx
      sy=info.sy
      ; forcing square
      dim = abs(dx-sx)<abs(dy-sy)
      dim = abs(dy-sy)<dim
      dx = (event.x gt sx) ? info.sx+dim-1:info.sx-dim+1
      dy = (event.y gt sy) ? info.sy+dim-1:info.sy-dim+1

      sx= min([info.sx,dx],max=dx)
      sy= min([info.sy,dy],max=dy)

      oldfrontendzoom = frontendzoom
      newfrontendzoom = round([sx/z.mgw+z.xmin,sy/z.mgw+z.ymin,dx/z.mgw+z.xmin,dy/z.mgw+z.ymin])

      if (abs(newfrontendzoom[0]-newfrontendzoom[2]) gt 1 ) and (abs(newfrontendzoom[1]-newfrontendzoom[3]) gt 1) then begin
        frontendzoom= newfrontendzoom
        case mousemode of
          1: begin
            case displaymode of
              0: meshwork_display, event, /smlm
              1: meshwork_display, event, /oft
              2: meshwork_display, event, /skeleton
              3: meshwork_display, event, /showmask
              4: meshwork_display, event, /maskoverlay
              5: meshwork_display, event, /ofoverlay
              6: meshwork_display, event, /skeleton
              7: begin
                meshwork_display, event, /smlm
                meshwork_display, event, /asters,/boundary
              end
              8: begin
                meshwork_display, event, /skeleton
                meshwork_display, event, /asters,/boundary
              end
            endcase
            frontendzoom=oldfrontendzoom
          end
          else:    case displaymode of
          0: meshwork_display, event, /smlm
          1: meshwork_display, event, /oft
          2: meshwork_display, event, /skeleton
          3: meshwork_display, event, /showmask
          4: meshwork_display, event, /maskoverlay
          5: meshwork_display, event, /ofoverlay
          6: meshwork_display, event, /skeleton
          7: begin
            meshwork_display, event, /smlm
            meshwork_display, event, /asters,/boundary
          end
          8:begin
            meshwork_display, event, /skeleton
            meshwork_display, event, /asters,/boundary
          end
        endcase
      endcase

    endif

    widget_control, event.top, set_uvalue=info,/no_copy
  end
endcase


end

pro meshwork_display_refresh, event
  displaymode = widget_info(Widget_Info(event.top, FIND_BY_UNAME='WID_DROPLIST_DISPLAY'),/droplist_select)
  case displaymode of
    0: meshwork_display, event, /smlm
    1: meshwork_display, event, /oft
    2: meshwork_display, event, /skeleton
    3: meshwork_display, event, /showmask
    4: meshwork_display, event, /maskoverlay
    5: meshwork_display, event, /ofoverlay
    6: meshwork_display, event, /skeleton
    7: begin
      meshwork_display, event, /smlm
      meshwork_display, event, /asters,/boundary
    end
    8: begin
      meshwork_display, event, /skeleton
      meshwork_display, event, /asters,/boundary
    end
  endcase
end

pro meshwork_color_refresh, event
  colorchoice = widget_info(Widget_Info(event.top, FIND_BY_UNAME='WID_DROPLIST_COLOR'),/droplist_select)
  colordroplist= [3,0,13,0,39,22] ;['Red Temperature # 3','Grayscale # 0','Rainbow # 13','Inverse Grayscale','Rainbow #39','Inverse Blue']
  case colorchoice of
    3: cgloadct, 0,/reverse
    5: cgloadct, 1,/reverse
    else: cgloadct, colordroplist[colorchoice]
  endcase

end




pro meshwork_display, event, reset=reset, frame=frame, averaged = averaged, in2x=in2x, out2x=out2x, maskoverlay=maskoverlay, showmask=showmask, smlm=smlm, $
  oft=oft, skeleton=skeleton,  skoverlay=skoverlay, ofoverlay=ofoverlay, nodes = nodes, idnode = idnode, idfilaments = idfilaments, selection= selection, $
  pinpoint = pinpoint, filaments =filaments, color=color, thick=thick, symsize=symsize, psym=psym, pores = pores, idpores = idpores, angles=angles, asters=asters, $
  boundary=boundary, idasters=idasters, q1=q1,q2=q2,q3=q3,q4=q4,quatre=quatre, noerase=noerase, lft=lft
  common display_info, def_w, wxsz,wysz, mousedown, infoinitial, frontendwindow, frontendzoom
  Common meshwork_dataset, properties, image_set, filamentmask, nodemask, node_table, pore_table, filament_table, angle_table

  displaymode = widget_info(Widget_Info(event.top, FIND_BY_UNAME='WID_DROPLIST_DISPLAY'),/droplist_select)

  if size(properties, /type) eq 0 then return
  if size(image_set, /type) eq 0 then return
  if ~keyword_set(psym) then psym = 9
  if ~keyword_set(symsize) then symsize = 4
  if ~keyword_set(color) then color = 'green'
  if ~keyword_set(thick) then thick = 2
  if keyword_set(quatre) then begin
    if keyword_set(q1) then position = [0.5,0.5,1.0,1.0]
    if keyword_set(q2) then position = [0.,0.5,0.5,1.0]
    if keyword_set(q3) then position = [0.,0.,0.5,0.5]
    if keyword_set(q4) then position = [0.5,0.,1.0,0.5]
  endif else position = [0,0,1,1]

  if keyword_set(reset) then begin
    zoomcoord = [0,0,properties.xpixels-1,properties.ypixels-1]
    frontendzoom = [0,0, properties.xpixels,properties.ypixels]
    return
  endif

  widget_control,widget_info(event.top, find_by_uname='WID_TABLE_FILAMENTS'), get_value=ftable
  z = meshwork_getzoomfactor()

  ;===asters

  if keyword_set(asters) then begin
    if size(image_set,/type) eq 0 then return
    if max(image_set[*,*,5]) eq 0 then return
    im = image_set[z.xmin:z.xmax,z.ymin:z.ymax,5]
    if max(im) eq 0 then return

    if keyword_set(boundary) then meshwork_plotboundary, event, binary= image_set[*,*,5] , /boundary, color='red',thick=2
    cgimage, bytscl(im), noerase=noerase, position = position, ctindex=38, transparent = 50,/reverse

    if keyword_set(idasters) then begin



      whereaster = where(im gt 0,/null)
      asteridlist = im[whereaster]
      asteridlist = asteridlist[uniq(asteridlist, sort(asteridlist))]
      numasterinbound = n_elements(asteridlist)

      for i = 0, numasterinbound-1 do begin
        wherethisaster = where(im eq asteridlist[i])
        idtext = string(asteridlist[i],format='(I5)')
        coord = array_indices(im, wherethisaster)
        cgtext,mean(coord[0,*])*z.mgw,mean(coord[1,*])*z.mgw,idtext,/device,color='orange'
      endfor
    endif

    return
  endif

  ;==pores
  if keyword_set(pores) or keyword_set(idpores) then begin
    if size(image_set,/type) eq 0 then return

    im = image_set[z.xmin:z.xmax,z.ymin:z.ymax,2]
    if max(im) eq 0 then return

    wherepore = where(im gt 0,/null)
    poreidlist = im[wherepore]
    poreidlist = poreidlist[uniq(poreidlist, sort(poreidlist))]
    numporeinbound = n_elements(poreidlist)
    print, 'plot ', numporeinbound, ' pores'
    if keyword_set(pores) and ~keyword_set(pinpoint) then cgimage, bytscl(im), /noerase, position = [0,0,1,1], ctindex=38, transparent = 50,/reverse

    if keyword_set(idpores) then begin
      for i = 0, numporeinbound-1 do begin
        wherethispore = where(im eq poreidlist[i])
        idtext = string(poreidlist[i],format='(I5)')
        coord = array_indices(im, wherethispore)
        cgtext,mean(coord[0,*])*z.mgw,mean(coord[1,*])*z.mgw,idtext,/device,color='orange'
      endfor
    endif

    if keyword_set(pinpoint) and n_elements(selection) gt 0 then begin
      for i = 0, n_elements(selection)-1 do begin
        wherethispore = where(im eq pore_table[0,selection[i]], count)
        thisporemask=im/0.
        thisporemask[wherethispore]=255
        if count gt 1 then begin
          coord = array_indices(im, wherethispore)
          cgplots,mean(coord[0,*])*z.mgw,mean(coord[1,*])*z.mgw,/device,psym=23,symsize=symsize,color='yellow',thick = thick
          cgimage, thisporemask,/noerase,transparent=80,/device,ctindex=18,/reverse
        endif else print,'Selected pore (s) not in current zoomed area'
      endfor
    endif

    return
  endif

  ;==filaments
  if keyword_set(filaments) or keyword_set(idfilaments) then begin
    if size(filamentmask,/type) eq 0 then return

    im = filamentmask[z.xmin:z.xmax,z.ymin:z.ymax]
    if max(im) eq 0 then return

    wherefil = where(im gt 0,/null)
    filidlist = im[wherefil]
    filidlist = filidlist[uniq(filidlist, sort(filidlist))]
    numfilinbound = n_elements(filidlist)
    print, 'plot ', numfilinbound, ' filaments'
    if keyword_set(filaments) and ~keyword_set(pinpoint) then cgimage, bytscl(im), /noerase, position = [0,0,1,1], ctindex=38, transparent = 50,/reverse

    if keyword_set(idfilaments) then begin
      for i = 0, numfilinbound-1 do begin
        wherethisfil = where(im eq filidlist[i])
        idtext = string(filidlist[i],format='(I5)')
        coord = array_indices(im, wherethisfil)
        cgtext,mean(coord[0,*])*z.mgw,mean(coord[1,*])*z.mgw,idtext,/device,color='blue'
      endfor
    endif

    if keyword_set(pinpoint) and n_elements(selection) gt 0 then begin
      for i = 0, n_elements(selection)-1 do begin
        wherethisfil = where(im eq ftable[0,selection[i]], count)
        if count gt 1 then begin
          coord = array_indices(im, wherethisfil)
          cgplots,mean(coord[0,*])*z.mgw,mean(coord[1,*])*z.mgw,/device,psym=23,symsize=symsize,color='yellow',thick = thick
        endif else print,'Selected filament (s) not in current zoomed area'
      endfor
    endif

    return
  endif

  ;==nodes
  if keyword_set(nodes) or keyword_set(idnode) then begin
    ;print, 'plot nodes'
    widget_control,widget_info(event.top, find_by_uname= 'WID_TABLE_NODES'), get_value = ntable
    nodex = ntable[1,*]
    nodey = ntable[2,*]

    whereinbound = where((nodex gt z.xmin) and (nodex lt z.xmax) and (nodey gt z.ymin) and (nodey lt z.ymax),count)
    print, 'in bound nodes:',count
    if keyword_set(nodes) and ~keyword_set(pinpoint) then begin
      if count gt 0 then       cgplots, (nodex[whereinbound]-z.xmin)*z.mgw, z.mgw*(nodey[whereinbound]-z.ymin),/device,psym=33,color='red'
    endif

    if keyword_set(pinpoint) and n_elements(selection) gt 0 then for i = 0,n_elements(selection)-1 do cgplots,(nodex[selection[i]]-z.xmin)*z.mgw,(nodey[selection[i]]-z.ymin)*z.mgw,/device,psym=psym,symsize=symsize,color=color,thick = thick

    if keyword_set(idnode) then begin
      if count gt 0 then for i = 0, count-1 do begin
        tposx = ((nodex[whereinbound[i]]+0.5)-z.xmin)*z.mgw
        tposy = ((nodey[whereinbound[i]]+0.5)-z.ymin)*z.mgw
        idtext = string(ntable[0,whereinbound[i]],format='(I5)')
        cgtext,tposx,tposy,idtext,/device,color=color
      endfor
    endif
    return
  endif

  ;==angles

  if keyword_set(angles) then begin

    widget_control,widget_info(event.top, find_by_uname= 'WID_TABLE_NODES'), get_value = ntable
    nodex = ntable[1,*]
    nodey = ntable[2,*]
    nodeid = reform(ntable[0,*])

    whereinbound = where((nodex gt z.xmin) and (nodex lt z.xmax) and (nodey gt z.ymin) and (nodey lt z.ymax),count)
    print, 'in bound nodes:',count
    ;    if keyword_set(nodes) and ~keyword_set(pinpoint) then begin
    ;      if count gt 0 then       cgplots, (nodex[whereinbound]-z.xmin)*z.mgw, z.mgw*(nodey[whereinbound]-z.ymin),/device,psym=33,color='red'
    ;    endif
    for i = 0, n_elements(whereinbound)-1 do meshwork_angles,event, selection=nodeid[whereinbound[i]] , /show, /idnode

    ;    if keyword_set(pinpoint) and n_elements(selection) gt 0 then for i = 0,n_elements(selection)-1 do    cgplots,(atable[2,selection[i]]+0.5-z.xmin)*z.mgw,(atable[3,selection[i]]+0.5-z.ymin)*z.mgw,$
    ;        /device,psym=psym,symsize=symsize,color=color,thick = thick
    return
  endif

  ;==general

  if keyword_set(in2x) then begin
    zoomold = meshwork_getzoomfactor()
    newxmin = (zoomold.xmin+zoomold.xrange*0.25)>0
    newxmax = (zoomold.xmax-zoomold.xrange*0.25)<(properties.xpixels)
    newymin = (zoomold.ymin+zoomold.yrange*0.25)>0
    newymax = (zoomold.ymax-zoomold.yrange*0.25)<(properties.ypixels)
    frontendzoom = [newxmin,newymin, newxmax,newymax]
  endif

  if keyword_set(out2x) then begin
    zoomold = meshwork_getzoomfactor()
    newxmin = (zoomold.xmin-zoomold.xrange*0.5)>0
    newxmax = (zoomold.xmax+zoomold.xrange*0.5)<(properties.xpixels)
    newymin = (zoomold.ymin-zoomold.yrange*0.5)>0
    newymax = (zoomold.ymax+zoomold.yrange*0.5)<(properties.ypixels)
    frontendzoom = [newxmin,newymin, newxmax,newymax]
  endif

  theframe =0
  if keyword_set(smlm) then theframe = 0
  if keyword_set(oft) then theframe = 1
  if keyword_set(skeleton) then theframe = 2
  if keyword_set(showmask) then theframe = 3
  if keyword_set(filaments) then theframe = 6
  if keyword_set(nodes) then theframe = 7
  if keyword_set(lft) then theframe = 8

  theframe=(theframe)>0
  z = meshwork_getzoomfactor()
  ;   help,z
  im = image_set[z.xmin:z.xmax,z.ymin:z.ymax,theframe]

  ;  minim = min(im, max = maxim)
  ;  maxim = maxim>(minim+1)

  imscl = congrid(im,z.mgw*z.xdim,z.mgw*z.ydim)

  if keyword_set(smlm) or keyword_set(oft) or keyword_set(lft) then cgimage, bytscl(imscl),position=position, noerase=noerase


  if keyword_set(showmask) then begin
    maskzoom = image_set[z.xmin:z.xmax,z.ymin:z.ymax, 3]
    cgImage, bytscl(~maskzoom, min =0, max = 1),ctIndex = xt, position = position,noerase=noerase
  endif

  if keyword_set(ofoverlay) then begin
    cgimage, image_set[z.xmin:z.xmax,z.ymin:z.ymax,0] ,/scale,position=position,noerase = noerase
    cgimage, image_set[z.xmin:z.xmax,z.ymin:z.ymax,1],/scale,position=position,/noerase, transparent=70, ctindex=3
    return
  endif

  if keyword_set(maskoverlay) then begin
    maskzoom = image_set[z.xmin:z.xmax,z.ymin:z.ymax, 3]
    cgImage, bytscl(~maskzoom, min =0, max = 1),ctIndex = xt, transparent = 70,position = position,/noerase
  endif

  if keyword_set(skeleton) then begin
    cgimage, bytscl(image_set[z.xmin:z.xmax,z.ymin:z.ymax,0]),position=position,noerase=noerase
    skelzoom = image_set[z.xmin:z.xmax,z.ymin:z.ymax, 2]
    cgImage, bytscl(~skelzoom, min =0, max = 1),ctIndex = 27, position = position,/noerase,/reverse,transparent=65
  endif
end


pro meshwork_plotboundary, event, binary= binaryimage , ellipse = ellipse, boundary=boundary, color = color, index =index, zoom=zoom, thick= thick, ctindex = ctindex
  common display_info, def_w, wxsz,wysz, mousedown, infoinitial, frontendwindow, frontendzoom
  common meshwork_dataset, properties, image_set, filamentmask, nodemask, node_table, pore_table, filament_table, angle_table, aster_table

  print, 'boundary..'
  if size(image_set,/type ) eq 0 then return
  if ~keyword_set(binaryimage) then return
  if size(binaryimage,/type ) eq 0 then return
  if max(binaryimage) eq 0 then return
  object = obj_new('Blob_Analyzer', binaryimage, mask=binaryimage)
  if ~obj_valid(object) then return
  numroi = object->NumberOfBlobs()
  print, 'numroi..',numroi
  if numroi lt 1 then return

  if keyword_set(thick) then tk=thick else tk = 2
  if keyword_set(ctindex) then cgloadct, ctindex

  width = (size(binaryimage,/dimensions))[0]
  height = (size(binaryimage,/dimensions))[1]

  z = meshwork_getzoomfactor()


  for i = 0, numroi -1 do begin
    stat = object->getstats(i)
    if keyword_set(index) then begin
      centerx = stat.center[0]
      centery = stat.center[1]
      x = centerx*z.mgw
      y = centery*z.mgw
      st = string(i+1,format='(I4)')
      cgtext,xo+z.mgw*x,yo+z.mgw*y, st,/device, color = color, charsize = 0.9
    endif
    if keyword_set(boundary) then begin
      perimx = (stat.perimeter_pts[0,*]-z.xmin+0.5)*z.mgw
      perimy = (stat.perimeter_pts[1,*]-z.ymin+0.5)*z.mgw
      cgplots, perimx,perimy, /device, color = color,thick = tk
      cgplots, perimx[0],perimy[0],/device,/continue,color = color, thick = tk
    endif
    if keyword_set(ellipse) then begin
      ;print, stat
      if ~((stat.mincol eq stat.maxcol) or (stat.minrow eq stat.maxrow)) then begin
        ell = object->fitellipse(i, center=center, axes=axes, orientation = orientation)
        cgplots, [xo,yo]+z.mgw*ell, color= color,/device
      endif
    endif
  endfor
end



pro meshwork_process_batch, event
  Common meshwork_dataset, properties, image_set, filamentmask, nodemask, node_table, pore_table, filament_table, angle_table, aster_table
  common display_info, def_w, wxsz,wysz, mousedown, infoinitial, frontendwindow, frontendzoom


  filename = Dialog_Pickfile(/read,/directory,get_path=fpath,title='Select a directory of _meshworks.sav files')
  if filename eq '' then return
  cd, fpath
  result = file_search(filename,'*_meshworks.sav',/fold_case)
  numfiles = n_elements(result)
  if n_elements(result) lt 1 then return

  print, 'Found..', numfiles, ' files:'
  print, result

  print,'Processing...'
  tbegin=systime(/seconds)
  for i =0, numfiles-1 do begin
    print, 'Loading files...:',result[i]
    meshwork_io,event,/loadsav,batch=result[i]
    meshwork_io_status,event,text = 'Loading files...:'+result[i]
    meshwork_display,event,/reset
    meshwork_display_refresh, event
    if min(image_set[*,*,2]) eq max(image_set[*,*,2]) then begin
      print,'No valid watershed for ...:', result[i]
      print,'Skipping...'
    endif else begin
      print,'Processing skeleton ...:', result[i]
      meshwork_io_status,event,text = 'Processing skeleton ...:'+result[i]
      meshwork_processing,event, /process,/verbose
      print,'Processing angle ...:', result[i]
      meshwork_io_status,event,text = 'Processing angle ...:'+result[i]
      meshwork_angles, event, /all
      meshwork_io,event,/savesav,batch=result[i]
    endelse
  endfor
  print,'Finish batch processing... time=',systime(/seconds)-tbegin,' sec.'
end

;function meshwork_watershed_relabel, watershedimage=watershedimage
;
;nregions = max(watershedimage)
;results = watershedimage*0
;
;poresize = dblarr(nregions)
;for i = 1, nregions do begin
;  wherethispore = where(watershedimage eq i,size)
;  poresize[i-1] = size
;endfor
;
;sizesort = sort(poresize)
;for i = 1, nregions do begin
;  oldporenumber = sizesort[i-1]+1
;  wherethispore = where(watershedimage eq oldporenumber,count)
;  results[wherethispore]=i
;endfor
;
;return, results
;
;end

function meshwork_watershed_relabel, watershedimage=watershedimage

  relabel = label_region(watershedimage ne 0,/all_neighbors)
  nregions = max(relabel)
  results = watershedimage*0

  poresize = dblarr(nregions)
  for i = 1, nregions do begin
    wherethispore = where(relabel eq i,size)
    poresize[i-1] = size
  endfor

  sizesort = sort(poresize)
  for i = 1, nregions do begin
    oldporenumber = sizesort[i-1]+1
    wherethispore = where(relabel eq oldporenumber,count)
    results[wherethispore]=i
  endfor

  return, results

end

pro meshwork_process_asters, event, update=update
  Common meshwork_dataset, properties, image_set, filamentmask, nodemask, node_table, pore_table, filament_table, angle_table, aster_table

  if keyword_set(update) then begin
    if size(image_set, /type) eq 0 then return
    if max(image_set[*,*,5]) eq 0 then return
    if max(image_set[*,*,0]) eq 0 then return
    if max(image_set[*,*,7]) eq 0 then return

    astermask = reform(image_set[*,*,5]) gt 0
    labelasters= label_region(astermask, /all_neighbors)
    numasters = max(labelasters)
    filamentmask = reform(image_set[*,*,7])

    cgimage, filamentmask gt 0,/scale,/noerase,position=[0,0,0.35,0.35]

    strel = replicate(1,5,5)
    astertable = dblarr(8,numasters)
    for i = 1, numasters do begin
      astertable[0,i-1] = i
      thisaster = where(labelasters eq i, count)
      thisastermask = astermask*0
      thisastermask[thisaster] = 1
      astertable[3,i-1] = count
      if count ge 1 then begin
        coord = array_indices(astermask, thisaster)
        astertable[4,i-1]=mean(image_set[coord[0,*],coord[1,*],0],/nan)
        astertable[5,i-1]=max(image_set[coord[0,*],coord[1,*],0])
        astertable[1,i-1] = mean(coord[0,*])
        astertable[2,i-1] = mean(coord[1,*])

        thisasterdilate = dilate(thisastermask,strel)
        cgimage, thisasterdilate, /scale,/noerase,position=[0,0,0.35,0.35]
        coorddilate = array_indices(astermask, thisasterdilate)
        overlapflmnt = filamentmask*thisasterdilate
        idoverlapflmnt = where(overlapflmnt gt 0, fcount)
        print, 'Aster #',i,' size:',count,' Filament overlap',fcount
        if fcount gt 0 then begin
          overlapflmnt = overlapflmnt[idoverlapflmnt]
          uniqflmnt = uniq(overlapflmnt,sort(overlapflmnt))
          uniqflmnt = overlapflmnt[uniqflmnt]
          numoverlapflmnt = n_elements(uniqflmnt)
          enclosedflmnt = 0
          if numoverlapflmnt gt 0 then begin
            for j = 0, numoverlapflmnt-1 do begin
              wherethisflmnt = where(filamentmask eq uniqflmnt[j], ffcount)
              if ffcount gt 1 then begin
                thisflmntmask = filamentmask*0
                thisflmntmask[wherethisflmnt] = 1
                if total(thisflmntmask or thisasterdilate) eq total(thisasterdilate) then begin
                  enclosedflmnt++
                  print,'Found enclosed filament: aster #',i
                endif
              endif
            endfor
          endif
          astertable[6,i-1]=numoverlapflmnt-enclosedflmnt
        endif
      endif
    endfor

    widget_control,widget_info(event.top, find_by_uname='WID_TABLE_ASTERS'), table_ysize=numasters
    widget_control,widget_info(event.top, find_by_uname='WID_TABLE_ASTERS'), row_labels=string(indgen(numasters)+1)
    widget_control,widget_info(event.top, find_by_uname='WID_TABLE_ASTERS'), set_value=astertable

    return
  endif

end

pro meshwork_process_skeleton, watershedimage=image, nodemask=nmask, ntable=ntable, filamentmask=fmask, verbose=verbose, $
  nnodes = numnodes, nfilaments = numfilaments, ftable=ftable, ptable=ptable, npores= numpores , smlmimage=smlmimage

  Common meshwork_dataset, properties, image_set, filamentmask, nodemask, node_table, pore_table, filament_table, angle_table, aster_table

  xdim = (size(image,/dimensions))[0]
  ydim = (size(image,/dimensions))[1]
  nmask = image*0
  mask = image eq 0
  fmask = image*0
  for i = 1, xdim-2 do begin
    for j = 1, ydim-2 do begin
      if mask[i,j] gt 0 then begin
        if  mask[i-1,j]+mask[i+1,j]+mask[i,j+1]+mask[i,j-1]  eq 2 then fmask[i,j] = 1 else nmask [i,j] = 1
      endif
    endfor
  endfor

  nmask = label_region(nmask,all_neighbors=0,/ulong)
  fmask = label_region(fmask,/ulong)
  numnodes = max(nmask)
  numfilaments = max(fmask)
  ntable = dblarr(21,numnodes)
  ftable = dblarr(14, numfilaments)
  cgimage, bytscl(nmask), position=[0,0,0.25,0.25],/noerase,ctindex = 13
  cgimage, bytscl(fmask), position=[0.25,0,0.5,0.25],/noerase,ctindex =3, transparent = 50


  for i = 0., numnodes-1 do begin
    wherethisnode = where(nmask eq (i+1), count)
    ;if count gt 1 then begin
    coordthisnode = array_indices(nmask, wherethisnode)
    ntable[0,i] = i+1
    ntable[1,i] = mean(coordthisnode[0,*])
    ntable[2,i] = mean(coordthisnode[1,*])
    ntable[3,i] = smlmimage[coordthisnode[0],coordthisnode[1]]
    ntable[4,i] = count
    if count gt 4 then print, 'Unusually large node:', i ,' count:', count

    if count eq 1 then filspaceneighborhood = fmask[coordthisnode[0]-1:coordthisnode[0]+1,coordthisnode[1]-1:coordthisnode[1]+1] $
    else filspaceneighborhood = fmask[min(coordthisnode[0,*])-1:max(coordthisnode[0,*])+1,min(coordthisnode[1,*])-1:max(coordthisnode[1,*])+1]

    wherefil = where(filspaceneighborhood ne 0,/null, count)
    thefil = filspaceneighborhood[wherefil]
    fl = thefil[uniq(thefil, sort(thefil))]
    for j = 0, n_elements(fl)-1 do ntable[5+j,i] = fl[j]

    if count eq 1 then  watershedneighborhood = image[coordthisnode[0]-1:coordthisnode[0]+1,coordthisnode[1]-1:coordthisnode[1]+1] $
    else watershedneighborhood = image[min(coordthisnode[0,*])-1:max(coordthisnode[0,*])+1,min(coordthisnode[1,*])-1:max(coordthisnode[1,*])+1]
    wherepore = where(watershedneighborhood ne 0,/null, count)
    thepores  = watershedneighborhood[wherepore]
    ws= thepores[uniq(thepores, sort(thepores))]
    for j = 0, n_elements(ws)-1 do ntable[13+j,i] = ws[j]
    ; endif
    if i mod 500 eq 0 then print, 'Node enumeration:', i
  endfor

  filblock = ntable[5:12,*]

  for i = 0., numfilaments-1 do begin
    wherethisfil = where(fmask eq (i+1), length)
    coordthisfil = array_indices(fmask, wherethisfil)
    ftable[0,i] = i+1
    if i mod 500 eq 0 then print, 'Filament enumeration:', i
    wherenodes = where(filblock eq (i+1), count)
    if count eq 0 then begin
      print, 'Error!!! node-count:',count,' i:',i , 'length:', length
      ftable[1,i] = -1
      ftable[2,i] = -1
      ftable[3,i] = -1
      ftable[4,i] = -1
      ftable[5,i] = -1
      ftable[6,i] = -1
      ftable[7,i] = -1
      ftable[8,i] = length
      ftable[13,i] = mean(smlmimage[wherethisfil],/nan)
    endif else if count ne 2 then begin
      nodei = array_indices(filblock, wherenodes[0])
      print, 'Error!!! node-count:',count,' i:',i , 'length:', length, ' node-ID:',ntable[0,nodei[1]]
      ftable[1,i] = ntable[0,nodei[1]]
      ftable[2,i] = -1
      ftable[3, i] = 1
      ftable[4,i] = ntable[1,nodei[1]]
      ftable[5,i] = ntable[2,nodei[1]]
      ftable[6,i] = -1
      ftable[7,i] = -1
      ftable[8,i] = length
      ftable[13,i] = mean(smlmimage[wherethisfil],/nan)
    endif else begin
      nodei = array_indices(filblock, wherenodes[0])
      ftable[1,i] = ntable[0,nodei[1]]
      nodef = array_indices(filblock, wherenodes[1])
      ftable[3,i] = 0
      ftable[2,i] = ntable[0,nodef[1]]
      ftable[4,i] = ntable[1,nodei[1]]
      ftable[5,i] = ntable[2,nodei[1]]
      ftable[6,i] = ntable[1,nodef[1]]
      ftable[7,i] = ntable[2,nodef[1]]
      ftable[8,i] = length
      ftable[13,i] = mean(smlmimage[wherethisfil],/nan)
    endelse
  endfor

  poreblock = ntable[13:20,*]
  numpores = max(image)
  print, 'Total pore number:',numpores
  ptable = dblarr(9, numpores)
  ptable[0,*] = findgen(numpores)+1
  S = REPLICATE(1, 3, 3)
  for i = 0, numpores-1 do begin
    wherethispore = where(image eq (i+1),count)
    coordthispore = array_indices(image, wherethispore)
    ptable[1,i] = mean(coordthispore[0,*])
    ptable[2,i] = mean(coordthispore[1,*])
    ptable[3,i] = count
    wherenodes = where(poreblock eq (i+1), vertices)
    ptable[4,i] = vertices
    ptable[6,i]=  mean(smlmimage[wherethispore],/nan)
    ptable[7,i]=  max(smlmimage[wherethispore],/nan)
    poremask= image*0
    poremask[wherethispore] = 1
    edgepore = dilate(poremask,s)
    edgepore[wherethispore]=0
    wherethisedge = where(edgepore, /null)
    ptable[8,i]=  mean(smlmimage[wherethisedge],/nan)
    if i mod 500 eq 0 then print, 'Pore enumeration:', i
  endfor


  if keyword_set(verbose) then print,' Filaments:', numfilaments
  if keyword_set(verbose) then print,' Nodes:', numnodes
  if keyword_set(verbose) then print,' Pores:', numpores
end

pro meshwork_processing,event, node=node , process=process, verbose=verbose
  Common meshwork_dataset, properties, image_set, filamentmask, nodemask, node_table, pore_table, filament_table, angle_table, aster_table

  if size(image_set, /type) eq 0 then return
  watershedimage = image_set[*,*,2]
  skeletonimage = image_set[*,*,3]
  if (min(watershedimage) eq max(watershedimage)) or (min(watershedimage) eq max(watershedimage)) then begin
    z = dialog_message("Invalid Watershed or Skeleton image!")
    return
  endif

  z = meshwork_getzoomfactor()
  imdim = size(watershedimage,/dimensions)
  xsize = imdim[0]
  ysize = imdim[1]

  if keyword_set(process) then begin
    print, 'Processing skeleton...'
    tbegin = systime(/seconds)
    meshwork_process_skeleton, watershedimage=watershedimage,nodemask=nodemask, ntable=ntable, filamentmask=filamentmask, verbose=verbose, $
      nnodes = numnodes, nfilaments = numfilaments, ftable=ftable, ptable =ptable, npores= numpores, smlmimage=image_set[*,*,0]
    ;[ 'pore-ID', 'Area','Edges','Edge-length','X-center','Y-center','Pore-Type']
    ;
    image_set[*,*,6] = nodemask
    image_set[*,*,7] = filamentmask

    ;help,ntable, numnodes
    for i = 0, numnodes -1 do begin
      cgplots, ntable[1,i]*z.mgw+z.xmin, z.ymin+z.mgw*ntable[2,i],color='red',psym=1,/device
    endfor

    widget_control,widget_info(event.top, find_by_uname='WID_TABLE_NODES'), table_ysize=numnodes
    widget_control,widget_info(event.top, find_by_uname='WID_TABLE_NODES'), row_labels=string(indgen(numnodes)+1)
    widget_control,widget_info(event.top, find_by_uname='WID_TABLE_NODES'), set_value=ntable
    widget_control,widget_info(event.top, find_by_uname='WID_TABLE_FILAMENTS'), table_ysize=numfilaments
    widget_control,widget_info(event.top, find_by_uname='WID_TABLE_FILAMENTS'), set_value=ftable
    widget_control,widget_info(event.top, find_by_uname='WID_TABLE_FILAMENTS'), row_labels=string(indgen(numfilaments)+1)
    widget_control,widget_info(event.top, find_by_uname='WID_TABLE_PORES'), table_ysize=numpores
    widget_control,widget_info(event.top, find_by_uname='WID_TABLE_PORES'), set_value=ptable
    widget_control,widget_info(event.top, find_by_uname='WID_TABLE_PORES'), row_labels=string(indgen(numpores)+1)

    node_table = ntable
    filament_table=ftable
    pore_table = ptable

    print,'Angle calculation..'
    meshwork_filaments, event, /all

    print,'Processing time:', systime(/seconds) - tbegin,' sec.'
    return
  endif



end

pro meshwork_oft, event, histogram = histogram, skeleton=skeleton, threshold=threshold, overlay=overlay, hminima=hminima, test=test, watershed=watershed, use=use, $
  normalize=normalize, full=full, nomask=nomask, matlab=matlab
  Common meshwork_dataset, properties, image_set, filamentmask, nodemask, node_table, pore_table, filament_table, angle_table, aster_table

  if size(image_set, /type) eq 0 then return
  widget_control,Widget_Info(event.top, FIND_BY_UNAME='WID_SLIDER_OFTTHRESHOLD'),get_value=oftthreshold
  widget_control,Widget_Info(event.top, FIND_BY_UNAME='WID_SLIDER_OFTTHRESHOLDMULTIPLIER'),get_value=oftthresholdmultiplier
  widget_control,Widget_Info(event.top, FIND_BY_UNAME='WID_SLIDER_OFTRADIUS'),get_value=oftradius
  widget_control,Widget_Info(event.top, FIND_BY_UNAME='WID_SLIDER_OFTSECTOR'),get_value=oftsector

  z = meshwork_getzoomfactor()

  if keyword_set(test) then begin
    testsmlm = reform(image_set[z.xmin:z.xmax,z.ymin:z.ymax,0])
    help, testsmlm
    matlab_OFT, image=testsmlm, radius=oftradius,sector=oftsector, OFT=OFT, LFT=LFT, Orientation = Orientation,/squarecrop,/padding, matlab=matlab
    cgimage, bytscl(OFT),/noerase,ctindex = 13,position=[0,0.8,0.2,1]
    cgimage, bytscl(LFT),/noerase,ctindex = 13,position=[0.2,0.8,0.4,1]
    cgimage, bytscl(Orientation),/noerase,ctindex = 13,position=[0.4,0.8,0.6,1]
    help, OFT
    return
  endif

  if keyword_set(full) then begin
    print,'Performing OFT...'
    meshwork_io_status,event, text = 'Performing OFT...'
    if (size(image_set,/dimensions))[2] lt 10 then begin
      old = image_set*1.
      image_set = dblarr((size(old,/dimensions))[0],(size(old,/dimensions))[1],10)
      image_set[*,*,0:(size(old,/dimensions))[2]-1]=old
    endif
    tbegin = systime(/seconds)
    matlab_OFT, image=reform(image_set[*,*,0]), radius=oftradius,sector=oftsector, OFT=OFT, LFT=LFT, Orientation = Orientation,/squarecrop,/padding, matlab=matlab
    cgimage, bytscl(OFT),/noerase,ctindex = 13,position=[0,0.8,0.2,1]
    cgimage, bytscl(LFT),/noerase,ctindex = 13,position=[0.2,0.8,0.4,1]
    cgimage, bytscl(Orientation),/noerase,ctindex = 13,position=[0.4,0.8,0.6,1]
    help, OFT
    image_set[*,*,1]=OFT
    image_set[*,*,8]=LFT
    image_set[*,*,9]=Orientation
    tstatus ='Finished OFT calculation:'+string(systime(/seconds)-tbegin,format='(F10.3)')+' sec.'
    print, tstatus
    meshwork_io_status,event, text = tstatus
    return
  endif

  if keyword_set(overlay) then begin
    thres =   oftthreshold*oftthresholdmultiplier
    oftimage = reform(image_set[*,*,1])
    thresoft = oftimage gt thres
    if total(thresoft) lt 1 then return
    maskzoom = thresoft[z.xmin:z.xmax,z.ymin:z.ymax]
    cgImage, bytscl(~maskzoom, min =0, max = 1),ctIndex = xt, transparent = 70,position = [0,0,1,1.],/noerase
    ;meshwork_plotboundary, event, binary= thresoft , /boundary, color = 'yellow'
    return
  endif

  if keyword_set(skeleton) then begin
    print, 'Skeleton from Vincent-Soulle Segmentation of Gauss-smooth OFT'
    ;thres =   oftthreshold*oftthresholdmultiplier
    oftimage = reform(image_set[*,*,0])
    watershed = watershed(gauss_smooth(oftimage,/edge_truncate),/long, connectivity=8)
    skeletonindex = where(watershed eq 0)
    skeleton = oftimage*0
    skeleton[skeletonindex] = 1
    if total(skeleton) lt 1 then return
    maskzoom = skeleton[z.xmin:z.xmax,z.ymin:z.ymax]
    cgImage, bytscl(~maskzoom, min =0, max = 1),ctIndex = xt, transparent = 70,position = [0,0,1,1.],/noerase
    ;meshwork_plotboundary, event, binary= thresoft , /boundary, color = 'yellow'
    image_set[*,*,2] = skeleton
    return
  endif

  if keyword_set(watershed) then begin
    mask = reform(image_set[z.xmin:z.xmax,z.ymin:z.ymax,4])
    if keyword_set(nomask) or max(mask) eq 0 then oftimage = reform(image_set[z.xmin:z.xmax,z.ymin:z.ymax,1]) else $
      oftimage = reform(image_set[z.xmin:z.xmax,z.ymin:z.ymax,1])*mask
    thres =   oftthreshold*oftthresholdmultiplier

    print,'H-minima+Meyer Watershed'
    print, 'h-minima of OFT/ Threshold value:', thres
    matlab_hmin, image=oftimage, h=thres, matlab=matlab, results=results
    cgimage, results,/scale,position=[0.,0.8,0.2,1],ctindex=13,/noerase
    cgimage, results,/scale,ctindex=13,/noerase,transparent=75
    matlab_watershed,image = results, matlab=matlab, results=watershedim
    ; skeletonindex = where(watershedim eq 0)
    ; skeleton = oftimage*0
    skeleton = watershedim eq 0 ;[skeletonindex] = 1
    ;help, results, watershedim
    cgimage, bytscl(skeleton),ctindex=27,transparent=70,position=[0,0,1,1.],/noerase
    cgimage, watershedim,/scale,position=[0.2,0.8,0.4,1],ctindex=3,/noerase,/reverse

    if keyword_set(use) then begin
      image_set[z.xmin:z.xmax,z.ymin:z.ymax,2] = watershedim
      image_set[z.xmin:z.xmax,z.ymin:z.ymax,3] = watershedim eq 0
    endif

    return
  endif

  if keyword_set(hminima) then begin

    oftimage = reform(image_set[z.xmin:z.xmax,z.ymin:z.ymax,1])
    thres =   oftthreshold*oftthresholdmultiplier
    print,'h-minima of OFT/ Threshold value:', thres
    matlab_hmin, image=oftimage, h=thres, matlab=matlab, results=results
    cgimage, results,/scale,position=[0.,0.8,0.2,1],ctindex=13,/noerase
    cgimage, results,/scale,ctindex=13,/noerase,transparent=75
    return
  endif

  if keyword_set(histogram) then begin
    oftimage = reform(image_set[*,*,1])
    nonzeroindex = where(image_set[*,*,1] gt 0, /null)
    print, 'Max:', max(image_set[*,*,1]),' Min:',min(image_set[*,*,1])
    if total(image_set[*,*,3]) eq 0 then begin
      cghistoplot,oftimage[nonzeroindex],axiscolorname = 'red', backcolorname='white',position = [0.1,0.1,0.6,0.6],nbins=100, histdata=histdata, title='Histogram of OFT (non zero-pixels)',/log
      ;otsulevel = cgotsu_threshold(oftimage[nonzeroindex],nbins = 100)
      otsulevel = cgotsu_threshold(oftimage[nonzeroindex])
    endif else begin
      maskindex = where(image_set[*,*,3] gt 0, /null)
      cghistoplot,oftimage[maskindex and nonzeroindex],axiscolorname = 'red', backcolorname='white',position = [0.1,0.1,0.6,0.6],nbins=100, histdata=histdata, title='Histogram of OFT (non zero-pixels)',/log
      ;otsulevel = cgotsu_threshold(oftimage[maskindex and nonzeroindex],nbins = 100)
      otsulevel = cgotsu_threshold(oftimage[maskindex and nonzeroindex])
    endelse
    print, 'Otsu-level (non-zero pixels):',otsulevel
    cgoplot,[otsulevel,otsulevel],[0,max(histdata)],/noerase,psym=33,color='blue'
    cgoplot,[otsulevel,otsulevel],[0,max(histdata)],/noerase,color='cyan'
    return
  endif

  if keyword_set(threshold) then begin
    oftimage = reform(image_set[*,*,1])
    print, 'Max:', max(image_set[*,*,1]),' Min:',min(image_set[*,*,1])
    threshold = image_threshold(oftimage,threshold=thres,/otsu)
    print,'Entired-image Threshold:', thres
    nonzeroindex = where(image_set[*,*,1] gt 0, /null)
    if total(image_set[*,*,3]) eq 0 then begin
      ;otsulevel = cgotsu_threshold(oftimage[nonzeroindex],nbins = 100)
      otsulevel = cgotsu_threshold(oftimage[nonzeroindex]) 
    endif else begin
      maskindex = where(image_set[*,*,3] gt 0, /null)
      ;otsulevel = cgotsu_threshold(oftimage[maskindex and nonzeroindex],nbins = 100)
      otsulevel = cgotsu_threshold(oftimage[maskindex and nonzeroindex])
    endelse
    print, 'Otsu-level (non-zero pixels):',otsulevel
    widget_control,Widget_Info(event.top, FIND_BY_UNAME='WID_SLIDER_OFTTHRESHOLD'),set_value=otsulevel
    return
  endif
end

function  meshwork_sensitivity_score, event, watershedimage = watershedimage, smlmimage= smlmimage

  wherefilaments = where(watershedimage eq 0,filamentlength)
  intensityfilaments = smlmimage[wherefilaments]

  wherepore = where(watershedimage ne 0,/null)
  s=replicate(1,3,3)
  poremaskerode = watershedimage*0
  poremaskerode[wherepore] = 1
  poremaskerode = erode(poremaskerode, s)
  whereporeerode = where(poremaskerode,/null)

  intensitypore = smlmimage[whereporeerode]

  return, {meanfilaments:mean(intensityfilaments,/nan), maxfilaments: max(intensityfilaments), meanpore: mean(intensitypore,/nan), maxpore: max(intensitypore), numpore:max(watershedimage), $
    filamentlength:filamentlength, minfilaments:min(intensityfilaments),stdfilaments:stddev(intensityfilaments)}

end

function meshwork_sensitivity_jaccard, event, groundtruth=groundtruth, image=image

  gmask = (groundtruth eq 0)
  input = (image eq 0)

  return, total(gmask and input)/total(gmask or input)

end

pro meshwork_sensitivity_analysis,event, load=load, segmentation=segmentation, summary=summary, montage=montage, overlay=overlay

  if keyword_set(load) then begin
    filename = Dialog_Pickfile(/read,get_path=fpath,filter=['*sensitivity.sav'],title='Select *meshworks.sav file to open')
    if filename eq '' then begin
      print,'filename not recognized', filename
      return
    endif
    cd,fpath

    print,'opening file: ', filename
    if strpos(filename,'sensitivity.sav') ne -1 then restore,filename=filename
  endif

  if keyword_set(segmentation) and keyword_set(summary) then begin
    cgimage,smlmimage, /scale,position=[0,0,0.25,0.25]
    print,'OFT-r',oftr
    print,'h-min',hmin

    porenumberheatmap = dblarr(n_elements(oftr), n_elements(hmin))
    totlfilheatmap   = dblarr(n_elements(oftr), n_elements(hmin))
    filstdevheatmap  = dblarr(n_elements(oftr), n_elements(hmin))
    meanporeintensityheatmap = dblarr(n_elements(oftr), n_elements(hmin))
    maxporeintensityheatmap = dblarr(n_elements(oftr), n_elements(hmin))
    filmeanheatmap = dblarr(n_elements(oftr), n_elements(hmin))
    filminheatmap = dblarr(n_elements(oftr), n_elements(hmin))
    print, jaccardarray
    print, unscaledthresarray
    for i = 0,n_elements(oftr)-1 do begin
      for j= 0,n_elements(hmin)-1 do begin
        thisscore=*scorearray[i,j]
        porenumberheatmap[i,j] = thisscore.numpore
        totlfilheatmap[i,j] = thisscore.filamentlength
        filstdevheatmap[i,j] = thisscore.stdfilaments
        meanporeintensityheatmap[i,j] = thisscore.meanpore
        maxporeintensityheatmap[i,j] = thisscore.maxpore
        filmeanheatmap[i,j] = thisscore.meanfilaments
        filminheatmap[i,j] = thisscore.minfilaments
      endfor
    endfor

    ;print, max(hmin)

    cgplot,oftr,hmin,/nodata,xstyle=1,yrange=[min(hmin),max(hmin)],position=[0.08,0.71,0.32,0.95], title='Number of Pores', xtitle='R',ytitle='rel.h-min'
    cgimage, porenumberheatmap, /scale, /noerase, position=[0.08,0.71,0.32,0.95],ctindex=3

    cgplot,oftr,hmin,/nodata,/noerase,xstyle=1,yrange=[min(hmin),max(hmin)],position=[0.08,0.38,0.32,0.62], title='Max Pore Intensity', xtitle='R',ytitle='rel.h-min'
    cgimage, maxporeintensityheatmap, /scale, /noerase, position=[0.08,0.38,0.32,0.62], ctindex =3

    cgplot,oftr,hmin,/nodata,/noerase,xstyle=1,yrange=[min(hmin),max(hmin)],position=[0.08,0.08,0.32,0.32], title='Jaccard Index (Filaments)', xtitle='R',ytitle='rel.h-min'
    cgimage, jaccardarray, /scale, /noerase, position=[0.08,0.08,0.32,0.32], ctindex =3
    ;   cgplot,oftr,hmin,/nodata,/noerase,xstyle=1,ystyle=1,position=[0.39,0.08,0.63,0.32], title='Number of Pores', xtitle='R',ytitle='rel.h-min'

    cgplot,oftr,hmin,/nodata,/noerase,xstyle=1,yrange=[min(hmin),max(hmin)],position=[0.39,0.38,0.63,0.62], title='Mean Pore intensity', xtitle='R',ytitle='rel.h-min'
    cgimage, meanporeintensityheatmap, /scale, /noerase, position=[0.39,0.38,0.63,0.62], ctindex = 3

    cgplot,oftr,hmin,/nodata,/noerase,xstyle=1,yrange=[min(hmin),max(hmin)],position=[0.39,0.71,0.63,0.95], title='Total Filament Lengths', xtitle='R',ytitle='rel.h-min'
    cgimage, totlfilheatmap, /scale, /noerase, position=[0.39,0.71,0.63,0.95], ctindex =3

    cgplot,oftr,hmin,/nodata,/noerase,xstyle=1,yrange=[min(hmin),max(hmin)],position=[0.71,0.08,0.95,0.32], title='Filament Min Intensity', xtitle='R',ytitle='rel.h-min'
    cgimage, filminheatmap, /scale, /noerase, position=[0.71,0.08,0.95,0.32],ctindex=3

    cgplot,oftr,hmin,/nodata,/noerase,xstyle=1,yrange=[min(hmin),max(hmin)],position=[0.71,0.38,0.95,0.62], title='Filament Mean Intensity', xtitle='R',ytitle='rel.h-min'
    cgimage, filmeanheatmap, /scale, /noerase, position=[0.71,0.38,0.95,0.62],ctindex=3

    cgplot,oftr,hmin,/nodata,/noerase,xstyle=1,yrange=[min(hmin),max(hmin)],position=[0.71,0.71,0.95,0.95], title='Filament Stdev Intensity', xtitle='R',ytitle='rel.h-min'
    cgimage, filstdevheatmap, /scale, /noerase, position=[0.71,0.71,0.95,0.95], ctindex =3
    ; watershedarray, scorearray, hminarray, otsuarray, unscaledthresarray, properties, z, smlmimage, oftr, hmin


    return
  endif


end

pro meshwork_sensitivity, event, segmentation=segmentation, enhancement=enhancement, geometry=geometry
  Common meshwork_dataset, properties, image_set, filamentmask, nodemask, node_table, pore_table, filament_table, angle_table, aster_table


  if keyword_set(enhancement) then begin
    widget_control,Widget_Info(event.top, FIND_BY_UNAME='WID_TEXT_OFTR'),get_value=oftr
    oftr = strsplit(oftr," ,",/extract,count=count)
    if n_elements(count) gt 1 then begin
      z=dialog_message('Only one line allowed..')
      return
    endif
    oftr = double(oftr)

    widget_control,Widget_Info(event.top, FIND_BY_UNAME='WID_TEXT_OFTTHETA'),get_value=ofttheta
    ofttheta = strsplit(ofttheta," ,",/extract,count=count)
    if n_elements(count) gt 1 then begin
      z=dialog_message('Only one line allowed..')
      return
    endif
    ofttheta = double(ofttheta)

    print, 'Sensitivity analysis...'
    print,oftr
    print, ofttheta

    widget_control,Widget_Info(event.top, FIND_BY_UNAME='WID_SLIDER_OFTTHRESHOLDMULTIPLIER'),get_value=oftthresholdmultiplier
    print,'Threshold multiplier..:',oftthresholdmultiplier

    z = meshwork_getzoomfactor()
    meshwork_goto, event, x=z.xc, y=z.yc, winsize=200
    meshwork_display_refresh,event
    filename = Dialog_Pickfile(/write,get_path=fpath,filter=['*sensitivity.sav'],title='Save sensitivity analysis results to comma-delimited file')
    if filename eq '' then return
    cd,fpath
    filename  =meshwork_addextension(filename,'_e_sensitivity.csv')
    print,'Output to be saved to...:',filename

    z = meshwork_getzoomfactor()
    ; print, z
    smlmimage = reform(image_set[z.xmin:z.xmax,z.ymin:z.ymax,0])
    groundtruth = reform(image_set[z.xmin:z.xmax,z.ymin:z.ymax,2])

    ;matlab_OFT, image=image, radius=radius,sector=sector, OFT=OFT, LFT=LFT, Orientation = Orientation, padding=padding, squarecrop = squarecrop
    panelx = n_elements(oftr)*1.
    panely = n_elements(ofttheta)*1.
    thresholdarray = dblarr(panelx,panely)
    cgimage, smlmimage,/scale,/noerase,position=[0,0,0.25,0.25]
    for i =0, n_elements(oftr)-1 do begin
      for j = 0, n_elements(ofttheta)-1 do begin
        matlab_OFT, image=smlmimage, radius=oftr[i],sector=ofttheta[j], OFT=OFT, LFT=LFT, Orientation = Orientation, /padding, matlab=matlab

        cgimage, oft,/scale,/noerase,position=[i/panelx,j/panely,(i+1)/panelx,(j+1)/panely],ctindex=13
        segmentimage = image_threshold(OFT,threshold=thres,/otsu)
        thresholdarray[i,j] = thres
        hminthres= thresholdarray[i,j]*oftthresholdmultiplier
        print, 'OFT-r:', oftr[i],' OFT-sector:',ofttheta[j],' Otsu threshold:',thresholdarray[i,j],' H-min threshold:', hminthres

        matlab_hmin, image=OFT, h=hminthres, matlab=matlab, results=results
        cgimage, results,/scale,/noerase,position=[i/panelx,j/panely,(i+1)/panelx,(j+1)/panely],ctindex=13
        ;help,results
        matlab_watershed,image = results, matlab=matlab, results=watershedim
        cgimage, watershedim,/scale,/noerase,position=[i/panelx,j/panely,(i+1)/panelx,(j+1)/panely],ctindex=13

        score = meshwork_sensitivity_score(smlmimage = smlmimage, watershedimage=watershedim)
        help, score
        print, 'Jaccard:',meshwork_sensitivity_jaccard(groundtruth=groundtruth, image=watershedim)
      endfor
    endfor

    return
  endif

  if keyword_set(segmentation) then begin
    widget_control,Widget_Info(event.top, FIND_BY_UNAME='WID_TEXT_OFTR_2'),get_value=oftr
    oftr = strsplit(oftr," ,",/extract,count=count)
    if n_elements(count) gt 1 then begin
      z=dialog_message('Only one line allowed..')
      return
    endif
    oftr = double(oftr)

    widget_control,Widget_Info(event.top, FIND_BY_UNAME='WID_TEXT_HMIN'),get_value=hmin
    hmin = strsplit(hmin," ,",/extract,count=count)
    if n_elements(count) gt 1 then begin
      z=dialog_message('Only one line allowed..')
      return
    endif
    hmin = double(hmin)

    print, 'Sensitivity analysis...'
    print, oftr
    print, hmin
    widget_control,Widget_Info(event.top, FIND_BY_UNAME='WID_SLIDER_OFTSECTOR'),get_value=oftsector
    print,'OFT sector..:',oftsector

    z = meshwork_getzoomfactor()
    ;print, z
    meshwork_goto, event, x=z.xc, y=z.yc, winsize=200
    meshwork_display_refresh,event
    filename = Dialog_Pickfile(/write,get_path=fpath,filter=['*sensitivity.sav'],title='Save sensitivity analysis results to comma-delimited file')
    if filename eq '' then return
    cd,fpath
    filename  =meshwork_addextension(filename,'_s_sensitivity.csv')
    savfilename = meshwork_addextension(filename,'.sav')
    print,'Output to be saved to...:',filename,' and ', savfilename

    z = meshwork_getzoomfactor()
    tbegin = systime(/seconds)


    smlmimage = reform(image_set[z.xmin:z.xmax,z.ymin:z.ymax,0])
    groundtruth = reform(image_set[z.xmin:z.xmax,z.ymin:z.ymax,2])
    panelx = n_elements(oftr)*1.
    panely = n_elements(hmin)*1.
    hminarray = dblarr(panelx,panely)
    otsuarray = dblarr(panelx,panely)
    unscaledthresarray = dblarr(panelx,panely)
    scorearray = ptrarr(panelx,panely)
    jaccardarray = dblarr(panelx,panely)
    cgimage, smlmimage,/scale,/noerase,position=[0,0,0.25,0.25]
    watershedarray = ptrarr(panelx,panely)
    header = strarr(n_elements(oftr)*n_elements(hmin))
    k = 0
    for i = 0, n_elements(oftr)-1 do begin
      for j = 0, n_elements(hmin)-1 do begin
        print,' Analyzing OFT-radius =',oftr[i],'  relative h-min=',hmin[j]
        matlab_OFT, image=smlmimage, radius=oftr[i],sector=oftsector, OFT=OFT, LFT=LFT, Orientation = Orientation, /padding, matlab=matlab
        cgimage, oft,/scale,/noerase,position=[i/panelx,j/panely,(i+1)/panelx,(j+1)/panely],ctindex=13
        ;help, oft
        segmentimage = image_threshold(OFT,threshold=thres,/otsu)
        otsuarray[i,j] = thres/max(oft)
        hminarray[i,j]= hmin[j]*otsuarray[i,j]
        unscaledthresarray[i,j] = thres*hminarray[i,j]
        ;print, otsuarray[i,j]
        oftnorm = double(oft)/max(oft)
        print, 'OFT-r:', oftr[i],' h-min relative:',hmin[j],' Otsu threshold:',otsuarray[i,j],' H-min relative threshold:', hminarray[i,j], 'Unscaled thres:',unscaledthresarray[i,j]
        matlab_hmin, image=OFTnorm, h=hminarray[i,j], matlab=matlab, results=results
        cgimage, results,/scale,/noerase,position=[i/panelx,j/panely,(i+1)/panelx,(j+1)/panely],ctindex=13
        ;help,results
        matlab_watershed, image = results, matlab=matlab, results=watershedim
        cgimage, watershedim,/scale,/noerase,position=[i/panelx,j/panely,(i+1)/panelx,(j+1)/panely],ctindex=31

        score = meshwork_sensitivity_score(smlmimage = smlmimage, watershedimage=watershedim)
        help, score
        jaccardarray[i,j] = meshwork_sensitivity_jaccard(groundtruth=groundtruth, image=watershedim)
        print, 'Jaccard:', jaccardarray[i,j]
        header[k]='r:'+string(oftr[i],format='(I8)') +'_h:'+string(hmin[j],format='(I8)')
        k++

        scorearray[i,j] = ptr_new(score)
        watershedarray[i,j] = ptr_new(watershedim)
      endfor
    endfor
    print,' Finish.. time=',systime(/seconds)-tbegin,' sec.'

    save, filename=savfilename, watershedarray, scorearray, hminarray, otsuarray, unscaledthresarray, jaccardarray, properties, z, smlmimage, oftr, hmin
    print,' Save finished..'


    return
  endif

  if keyword_set(geometry) then begin
    print, 'Sensitivity analysis...'
    widget_control,Widget_Info(event.top, FIND_BY_UNAME='WID_TEXT_TANGENT'),get_value=tangentval
    tangentval = strsplit(tangentval," ,",/extract,count=count)
    if n_elements(count) gt 1 then begin
      z=dialog_message('Only one line allowed..')
      return
    endif
    tangentval = double(tangentval)

    widget_control,Widget_Info(event.top, FIND_BY_UNAME='WID_TEXT_BOXCAR'),get_value=boxcarval
    boxcarval = strsplit(boxcarval," ,",/extract,count=count)
    if n_elements(count) gt 1 then begin
      z=dialog_message('Only one line allowed..')
      return
    endif
    boxcarval = double(boxcarval)

    print, tangentval
    print, boxcarval

    filename = Dialog_Pickfile(/write,get_path=fpath,filter=['*.csv','*.txt'],title='Save sensitivity analysis results to comma-delimited file')
    if filename eq '' then return
    cd,fpath
    filename  =meshwork_addextension(filename,'_g_sensitivity.csv')
    print,'Output to be saved to...:',filename

    tbegin = systime(/seconds)

    interfangleresults = dblarr(n_elements(tangentval)*n_elements(boxcarval), 361)
    header = strarr(n_elements(tangentval)*n_elements(boxcarval))
    k = 0
    for i = 0, n_elements(tangentval)-1 do begin
      for j = 0, n_elements(boxcarval)-1 do begin
        print,' Analyzing tangent =',tangentval[i],'  boxcar=',boxcarval[j]
        meshwork_filaments, event, /all, boxcar=boxcarval[j], tangent=tangentval[i]
        meshwork_angles, event, /all
        widget_control,widget_info(event.top, find_by_uname='WID_TABLE_ANGLES'), get_value=angle_table
        cghistoplot, angle_table[6,*], binsize=1,mininput=0,maxinput=360, color='red',position=[0.37,0.08,0.55,0.34], title='Interfilament Angle (optim./degree)',histdata=histdata, locations= loc
        interfangleresults[k,*] = histdata
        header[k]='t:'+string(tangentval[i],format='(I8)') +'_b:'+string(boxcarval[j],format='(I8)')
        k++
      endfor
    endfor
    print,' Finish.. time=',systime(/seconds)-tbegin,' sec.'

    write_csv,filename,interfangleresults,header=header

    return
  endif


end

pro meshwork_angles, event, selection=selection , show=show, all=all, idnodeid=idnode
  Common meshwork_dataset, properties, image_set, filamentmask, nodemask, node_table, pore_table, filament_table, angle_table, aster_table

  if size(properties, /type) eq 0 then return
  if size(image_set, /type) eq 0 then return
  if size(filament_table, /type) eq 0 then return
  arrowlength = 50

  z = meshwork_getzoomfactor()

  nodeid = reform(node_table[0,*])
  filamentid = reform(filament_table[0,*])

  if keyword_set(all) then begin
    print,' Calculating inter-filament angle..'
    tbegin = systime(/seconds)
    angleid = 1.
    anglestructptrarr = ptrarr(n_elements(nodeid))
    for i = 0, n_elements(nodeid)-1 do begin
      thisx = node_table[1,i]
      thisy = node_table[2,i]
      fportion = reform(node_table[5:12,i])
      index = where(fportion ne 0, fcount)

      flmntlist = fportion[index]
      fitangle = dblarr(fcount)
      grangle = dblarr(fcount)
      fillength = dblarr(fcount)
      optangle = dblarr(fcount)

      for j = 0, fcount-1 do begin
        wherethisfil = where(filamentmask eq flmntlist[j])
        coord = array_indices(filamentmask, wherethisfil)

        thisfilindex = where(filamentid eq flmntlist[j], count)
        if count ne 1 then print, 'Error!!'
        if filament_table[1,thisfilindex] eq node_table[0,i] then begin
          fitangle[j] = filament_table[9, thisfilindex]
          grangle[j] = filament_table[11, thisfilindex]
          fillength[j] = filament_table[8, thisfilindex]
        endif else if filament_table[2,thisfilindex] eq  node_table[0,i] then begin
          fitangle[j] = filament_table[10, thisfilindex]
          grangle[j] = filament_table[12, thisfilindex]
          fillength[j] = filament_table[8, thisfilindex]
        endif else begin
          print, 'Error!  Invalid filament..',  node_table[0,i], filament_table[1:2,thisfilindex]
        endelse

      endfor

      reorder = sort(fitangle)
      flmntlist = flmntlist[reorder]
      fitangle = fitangle[reorder]
      grangle = grangle[reorder]
      optangle = meshwork_mixangle(a1=grangle, a2=fitangle)
      ; optangle = 0.5*(fitangle+grangle)

      anglestruct = []

      interfangle = meshwork_diff_angle(a1=fitangle[0],a2=fitangle[-1],/cw);360-abs(fitangle[0]-fitangle[-1])
      dinterfangle = meshwork_diff_angle(a1=grangle[0],a2=grangle[-1],/cw);(360-abs(grangle[0]-grangle[-1])) mod 360
      optinterfangle = meshwork_diff_angle(a1=optangle[0],a2=optangle[-1],/cw);(360-abs(optangle[0]-optangle[-1])) mod 360

      ;      interfangle = (360-abs(fitangle[0]-fitangle[0-1])) mod 360
      ;      dinterfangle = (360-abs(grangle[0]-grangle[-1])) mod 360
      ;      optinterfangle = (360-abs(optangle[0]-optangle[-1])) mod 360

      anglestruct = [anglestruct, {angleid:angleid,anglecode:fcount,x:thisx, y: thisy, nodeid:node_table[0,i] , dangle:dinterfangle, oangle: optinterfangle,$
        fil1id: flmntlist[0],fil2id:flmntlist[-1],angle:interfangle, fil1length:fillength[0],fil2length:fillength[-1]}]
      angleid++
      for j = 1, fcount-1 do  begin
        interfangle = meshwork_diff_angle(a1=fitangle[j],a2=fitangle[j-1],/cw)
        ; interfangle = abs(fitangle[j]-fitangle[j-1]) mod 180
        dinterfangle = meshwork_diff_angle(a1=grangle[j],a2=grangle[j-1],/cw)
        ;dinterfangle = abs(grangle[j]-grangle[j-1]) mod 180
        optinterfangle = meshwork_diff_angle(a1=optangle[j],a2=optangle[j-1],/cw)
        ;optinterfangle = abs(optangle[j]-optangle[j-1]) mod 180
        anglestruct = [anglestruct, {angleid:angleid,anglecode:fcount,x:thisx, y: thisy, nodeid:node_table[0,i] , dangle:dinterfangle, oangle: optinterfangle, $
          fil1id: flmntlist[j],fil2id:flmntlist[j-1],angle:interfangle, fil1length:fillength[j],fil2length:fillength[j-1]}]
        angleid++
      endfor

      anglestructptrarr[i] = ptr_new(anglestruct)

      if i mod 500 eq 0 then print, 'Angle enumeration @ node:', i
    endfor

    print, 'Finish enumerating angle..', angleid, ' angles found'

    angle_table = dblarr(12 ,angleid-1)
    k=0.
    for i = 0, n_elements(nodeid)-1 do begin
      thisnodeanglestruct = *anglestructptrarr[i]
      for j = 0, n_elements(thisnodeanglestruct)-1 do begin
        angle_table[0,k] = thisnodeanglestruct[j].angleid
        angle_table[1,k] = thisnodeanglestruct[j].nodeid
        angle_table[2,k] = thisnodeanglestruct[j].x
        angle_table[3,k] = thisnodeanglestruct[j].y
        angle_table[4,k] = thisnodeanglestruct[j].angle
        angle_table[5,k] = thisnodeanglestruct[j].dangle
        angle_table[6,k] = thisnodeanglestruct[j].oangle
        angle_table[7,k] = thisnodeanglestruct[j].fil1id
        angle_table[8,k] = thisnodeanglestruct[j].fil2id
        angle_table[9,k] = thisnodeanglestruct[j].fil1length
        angle_table[10,k] = thisnodeanglestruct[j].fil2length
        angle_table[11,k] = thisnodeanglestruct[j].anglecode
        k++
      endfor
    endfor
    widget_control,widget_info(event.top, find_by_uname='WID_TABLE_ANGLES'), table_ysize=angleid-1
    widget_control,widget_info(event.top, find_by_uname='WID_TABLE_NODES'), row_labels=string(indgen(angleid-1)+1)
    widget_control,widget_info(event.top, find_by_uname='WID_TABLE_ANGLES'), set_value=angle_table
    print, 'Finish angle calculation... Time elapsed:',systime(/seconds)-tbegin,' sec.'
    return
  endif

  if keyword_set(show) then begin
    if keyword_set(idnode) then begin
      whichrow = where(node_table[0,*] eq selection[0], count)
      if count lt 1 then begin
        print, 'Error!'
        z=dialog_message('Error.. Node not found..')
        return
      endif
      whichrow = whichrow[0]
      print, 'Row #..', whichrow
    end else whichrow = selection[0]

    thisx = node_table[1,whichrow]
    thisy = node_table[2,whichrow]
    thisncode = node_table[4,whichrow]
    cgplots, (thisx+0.5-z.xmin)*z.mgw, z.mgw*(thisy+0.5-z.ymin),/device,psym=33,color='red'

    fportion = reform(node_table[5:12,whichrow])
    index = where(fportion ne 0, fcount)
    flmntlist = fportion[index]
    fitangle = dblarr(fcount)
    grangle = dblarr(fcount)
    optangle = dblarr(fcount)
    fillength = dblarr(fcount)

    for i = 0, fcount-1 do begin
      wherethisfil = where(filamentmask eq flmntlist[i])
      coord = array_indices(filamentmask, wherethisfil)
      idtext = string(flmntlist[i],format='(I5)')
      ;print, idtext
      cgtext,(mean(coord[0,*])-z.xmin)*z.mgw,(mean(coord[1,*])-z.ymin)*z.mgw,idtext,/device,color='blue'
      cgplots, (coord[0,*]-z.xmin)*z.mgw,(coord[1,*]-z.ymin)*z.mgw, psym=3, symsize = 1, color='red',/device

      thisfilindex = where(filamentid eq flmntlist[i], count)
      if count ne 1 then print, 'Error!!'
      if filament_table[1,thisfilindex] eq node_table[0,whichrow] then begin
        fitangle[i] = filament_table[9, thisfilindex]
        grangle[i] = filament_table[11, thisfilindex]
        fillength[i] = filament_table[8, thisfilindex]
      endif else if filament_table[2,thisfilindex] eq  node_table[0,whichrow] then begin
        fitangle[i] = filament_table[10, thisfilindex]
        grangle[i] = filament_table[12, thisfilindex]
        fillength[i] = filament_table[8, thisfilindex]
      endif else begin
        print, 'Error!  Invalid filament..',  node_table[0,whichrow], filament_table[1:2,thisfilindex]
      endelse

    endfor

    reorder = sort(fitangle)
    flmntlist = flmntlist[reorder]
    fitangle = fitangle[reorder]
    grangle = grangle[reorder]
    optangle = meshwork_mixangle(a1=grangle, a2=fitangle)

    print, 'fil-id:', flmntlist
    print, 'fit-angle:', fitangle
    print, 'direct-angle:',grangle
    print, 'opt.angle:',optangle
    print, ''

    offset= 0.5
    if thisncode ge 4 then offset = 1

    anglestruct = []
    cgarrow, (thisx+offset-z.xmin)*z.mgw,z.mgw*(thisy+offset-z.ymin),$
      (thisx+offset-z.xmin)*z.mgw+arrowlength*0.75*cos(fitangle[0]*!dtor),z.mgw*(thisy+offset-z.ymin)+arrowlength*0.75*sin(fitangle[0]*!dtor),color='cyan',/device

    interfangle = dblarr(fcount)
    dinterfangle = dblarr(fcount)
    optinterfangle = dblarr(fcount)

    interfangle[0] = meshwork_diff_angle(a1=fitangle[0],a2=fitangle[-1],/cw);360-abs(fitangle[0]-fitangle[-1])
    dinterfangle[0] = meshwork_diff_angle(a1=grangle[0],a2=grangle[-1],/cw);(360-abs(grangle[0]-grangle[-1])) mod 360
    optinterfangle[0] = meshwork_diff_angle(a1=optangle[0],a2=optangle[-1],/cw);(360-abs(optangle[0]-optangle[-1])) mod 360
    ;    print,'Fit-Inter-filament-angle:',interfangle
    ;    print,'direct-Inter-filament-angle:',dinterfangle
    ;    print,'optim-Inter-filament-angle:',optinterfangle

    cgarrow, (thisx+offset-z.xmin)*z.mgw,z.mgw*(thisy+offset-z.ymin),$
      (thisx+offset-z.xmin)*z.mgw+arrowlength*cos(grangle[0]*!dtor),z.mgw*(thisy+offset-z.ymin)+arrowlength*sin(grangle[0]*!dtor),color='yellow',/device, hthick=0.5
    cgarrow, (thisx+offset-z.xmin)*z.mgw,z.mgw*(thisy+offset-z.ymin),$
      (thisx+offset-z.xmin)*z.mgw+arrowlength*1.2*cos(optangle[0]*!dtor),z.mgw*(thisy+offset-z.ymin)+arrowlength*1.2*sin(optangle[0]*!dtor),color='orange',/device, hthick=0.5

    for i = 1, fcount-1 do  begin
      cgarrow, (thisx+offset-z.xmin)*z.mgw,z.mgw*(thisy+offset-z.ymin),$
        (thisx+offset-z.xmin)*z.mgw+arrowlength*0.75*cos(fitangle[i]*!dtor),z.mgw*(thisy+offset-z.ymin)+arrowlength*0.75*sin(fitangle[i]*!dtor),color='cyan',/device

      interfangle[i] = meshwork_diff_angle(/cw,a1=fitangle[i],a2=fitangle[i-1]);360-abs(fitangle[0]-fitangle[-1])
      dinterfangle[i] = meshwork_diff_angle(/cw,a1=grangle[i],a2=grangle[i-1]);(360-abs(grangle[0]-grangle[-1])) mod 360
      optinterfangle[i] = meshwork_diff_angle(/cw,a1=optangle[i],a2=optangle[i-1]);(360-abs(optangle[0]-optangle[-1])) mod 360

      ;      interfangle[i] = abs(fitangle[i]-fitangle[i-1]) mod 180
      ;      dinterfangle[i] = abs(grangle[i]-grangle[i-1]) mod 180
      ;      optinterfangle[i] = abs(optangle[i]-optangle[i-1]) mod 180

      cgarrow, (thisx+offset-z.xmin)*z.mgw,z.mgw*(thisy+offset-z.ymin),$
        (thisx+offset-z.xmin)*z.mgw+arrowlength*cos(grangle[i]*!dtor),z.mgw*(thisy+offset-z.ymin)+arrowlength*sin(grangle[i]*!dtor),color='yellow',/device, hthick=0.5
      cgarrow, (thisx+offset-z.xmin)*z.mgw,z.mgw*(thisy+offset-z.ymin),$
        (thisx+offset-z.xmin)*z.mgw+arrowlength*1.2*cos(optangle[i]*!dtor),z.mgw*(thisy+offset-z.ymin)+arrowlength*1.2*sin(optangle[i]*!dtor),color='orange',/device, hthick=0.5
    endfor

    print,'Fit-Inter-filament-angle:',interfangle
    print,'direct-Inter-filament-angle:',dinterfangle
    print,'optim-Inter-filament-angle:',optinterfangle

    return
  endif

end


pro meshwork_asters, event, show=show, selection=selection, montage=montage, dimension=dimension
  Common meshwork_dataset, properties, image_set, filamentmask, nodemask, node_table, pore_table, filament_table, angle_table, aster_table

  if size(properties, /type) eq 0 then return
  if size(image_set, /type) eq 0 then return
  if ~keyword_set(dimension) then dimension=100.
  widget_control,widget_info(event.top, find_by_uname='WID_TABLE_ASTERS'), get_value=aster_table

  if keyword_set(show) then begin
    if n_elements(selection) lt 1 then return
    meshwork_imageset_retrieve, x =aster_table[1,selection[0]], y=aster_table[2,selection[0]], /smlm, image=im
    cgimage, im,/scale,/noerase,position=[0,0,0.25,0.25]
    cgplots, 0.125,0.125, psym=2,color='green',symsize=1,/normal,thick=1
    return
  endif

  if keyword_set(montage) then begin
    meshwork_imageset_retrieve, x =aster_table[1,0], y=aster_table[2,0], /smlm, image=im, dimx=dimx, dimy=dimy
    montarray = dblarr(dimx,dimy,n_elements(aster_table[0,*]))
    for i = 0, n_elements(aster_table[0,*])-1 do begin
      meshwork_imageset_retrieve, x =aster_table[1,i], y=aster_table[2,i], /smlm, image=im
      montarray[*,*,i] = im
    endfor
    montageim = meshwork_makemontage(image_stack=montarray, /auto) ;, x=x, y=y,transpose=transpose, column = column,wxsize=wxsize,wysize=wysize, panels=panels
    cgimage, montageim,/scale

    return
  endif

end

pro meshwork_filaments, event, selection=selection , show=show, boxcar=boxcar, tangent=tangent, all=all, cutoff=cutoff
  Common meshwork_dataset, properties, image_set, filamentmask, nodemask, node_table, pore_table, filament_table, angle_table, aster_table

  if size(properties, /type) eq 0 then return
  if size(image_set, /type) eq 0 then return
  if size(filament_table, /type) eq 0 then return
  if ~keyword_set(boxcar) then boxcar = 3
  if ~keyword_set(tangent) then tangent = 5
  if ~keyword_set(cutoff) then cutoff = 15
  z = meshwork_getzoomfactor()

  filamentid = reform(filament_table[0,*])

  if keyword_set(all) then begin
    for i = 0, n_elements(filamentid)-1 do begin
      wherethisfil = where(filamentmask eq filamentid[i], length)
      coord = array_indices(filamentmask, wherethisfil)
      filx = reform(coord[0,*])
      fily = reform(coord[1,*])
      xi = filament_table[4,i]
      xf = filament_table[6,i]
      yi = filament_table[5,i]
      yf = filament_table[7,i]
      endendlength = sqrt(abs(xf-xi)^2+abs(yf-yi)^2)
      tortuosity = (length+1)/endendlength
      fit = meshwork_filaments_sort(xi=xi,yi=yi,xf=xf,yf=yf, xpoints=filx,ypoints=fily,boxcar=boxcar, cutoff=cutoff, tangent=tangent)
      filament_table[9,i] = fit.iangle
      filament_table[10,i] = fit.fangle
      filament_table[11,i] = fit.igrangle
      filament_table[12,i] = fit.fgrangle

      if i mod 500 eq 0 then print, 'Angle enumeration:', i
    endfor
    widget_control,widget_info(event.top, find_by_uname='WID_TABLE_FILAMENTS'), set_value=filament_table
    return
  endif

  if keyword_set(show) then begin
    wherethisfil = where(filamentmask eq filamentid[selection[0]], length)
    coord = array_indices(filamentmask, wherethisfil)
    cgimage, bytscl(bytarr(10,10)+1.), position = [0,0,0.35,0.35],/noerase
    filx = reform(coord[0,*])
    fily = reform(coord[1,*])
    xi = filament_table[4,selection[0]]
    xf = filament_table[6,selection[0]]
    yi = filament_table[5,selection[0]]
    yf = filament_table[7,selection[0]]
    if (xi-xf)*(yi-yf) eq 0 then cgplot,[filx,xi,xf ], [fily,yi,yf], position = [0.08,0.08,0.32,0.32],/noerase, psym=1, color='red', axiscolor='black', background = 'white',xstyle=1,ystyle=1 else $
      cgplot,[filx,xi,xf ], [fily,yi,yf], position = [0.08,0.08,0.32,0.32],/noerase, psym=1, color='red', axiscolor='black', background = 'white',/isotropic


    cgplots, xi, yi,color='red',psym=33, symsize=3
    cgplots, xf, yf,color='blue',psym=33, symsize=3
    cgplots,  [xi, xf], [yi, yf],color='black',linestyle = 2
    endendlength = sqrt(abs(xf-xi)^2+abs(yf-yi)^2)
    tortuosity = (length+2)/endendlength
    directangle = atan((yf-yi)/(xf-xi))/!DTOR

    fit = meshwork_filaments_sort(xi=xi,yi=yi,xf=xf,yf=yf, xpoints=filx,ypoints=fily,boxcar=boxcar, cutoff=cutoff)
    ;    xsm = ts_smooth([xi, fit.x, xf],boxcar,/double)
    ;    ysm = ts_smooth([yi, fit.y, yf],boxcar,/double)
    cgplots, fit.xsmooth,fit.ysmooth,color='orange', thick=2
    cgarrow, xi,yi,xi-cos(fit.iangle*!dtor)*tangent,yi-sin(fit.iangle*!dtor)*tangent,/data,color='cyan'
    cgarrow, xf,yf,xf+cos(fit.fangle*!dtor)*tangent,yf+sin(fit.fangle*!dtor)*tangent,/data,color='cyan'


    print, 'Length:', length,' end-to-end:', endendlength, ' tortuosity:', tortuosity,' i-directangle(deg):',fit.igrangle,' f-directangle(deg):',fit.fgrangle, 'fit-angle-i:',fit.iangle,' fit-angle-f:',fit.fangle
    print, 'i-Quad:', fit.iquadrant, ' f-Quad:',fit.fquadrant

    return
  endif

end

pro meshwork_quantify, event, batch=batch, export = export, noerase=noerase, pixelsize=pixelsize, cell=cell, pooled=pooled,noheader=noheader
  Common meshwork_dataset, properties, image_set, filamentmask, nodemask, node_table, pore_table, filament_table, angle_table, aster_table
  common display_info, def_w, wxsz,wysz, mousedown, infoinitial, frontendwindow, frontendzoom

  if ~keyword_set(pixelsize) then pixelsize =0.02
  if ~keyword_set(noerase) then cgerase,'white' ;cgimage,bytscl(bytarr(1024,1024)+1)

  nodeintensitybinsize = 20.
  flmtintensitybinsize = 20.
  poreareabinsize =  0.01
  anglebinsize = 5.
  flmntlengthbinsize = 0.1
  flmnttortbinsize = 0.1

  if ~keyword_set(batch) then begin
    if size(image_set,/type) eq 0 then return
    widget_control, widget_info(event.top,find_by_uname='WID_TABLE_PORES'), get_value=pore_table
    widget_control, widget_info(event.top,find_by_uname='WID_TABLE_NODES'), get_value=node_table
    widget_control, widget_info(event.top,find_by_uname='WID_TABLE_FILAMENTS'), get_value=filament_table
    widget_control, widget_info(event.top,find_by_uname='WID_TABLE_ANGLES'), get_value=angle_table

    nid = node_table[0,*]
    fid = filament_table[0,*]
    pid = pore_Table[0,*]
    aid = angle_table[0,*]
    pcode = pore_table[5,*]
    okpore = where(pcode ge 0, countgoodpore)

    t1a = 'Total # nodes:'+ string(n_elements(nid), format = '(I10)')+ ' Total # filaments:'+ string(n_elements(fid),format ='(I10)')
    t1b = 'Total # pores:'+string(n_elements(pid), format='(I10)')+' Total # of angles:'+string(n_elements(aid), format='(I10)')
    meshwork_io_status,event,text=t1

    cgtext, 0.025,0.98, properties.datafile,color='blue',/normal,charsize =1.5
    cgtext, 0.58,0.95, t1a,color='blue',/normal
    cgtext, 0.58,0.95-0.015, t1b,color='blue',/normal

    nodeintensity = reform(node_table[3,*])
    cghistoplot, nodeintensity, binsize=nodeintensitybinsize, /noerase, color='red',position=[0.075,0.08,0.30,0.34], title='SMLM intensity at nodes (avg)',mininput=0,loc=xlocnode
    help, xlocnode
    t2a = 'Mean Intensity:' +string(mean(nodeintensity,/nan), format='(F6.2)')
    t2b = 'Min Intensity:' +string(min(nodeintensity), format='(F6.2)')
    t2c = 'Max Intensity:' +string(max(nodeintensity), format='(F6.2)')
    cgtext, 0.01,0.01,t2b,color='blue',/normal
    cgtext, 0.01,0.0235,t2c, color='blue',/normal
    cgtext, 0.01,0.0235+0.0135,t2a, color='blue',/normal

    contourlength = filament_table[8,*]
    vectorlength = sqrt((filament_table[4,*]-filament_table[6,*])^2+ (filament_table[5,*]-filament_table[7,*])^2)
    tortuosity = (contourlength+2)/vectorlength
    avgintesity = filament_table[13,*]
    cghistoplot, contourlength*pixelsize, binsize=flmntlengthbinsize, /noerase, color='red',position=[0.42,0.08,0.64,0.34], title='Fil-Seg contour-length (micron)',xtitle = 'micron', mininput=0, $
      loc=x_fil, histdata=y_fil
    cghistoplot, tortuosity, binsize=flmnttortbinsize, /noerase, color='red',position=[0.75,0.08,0.97,0.34], title='Fil-Seg Tortuosity',mininput=1, loc=xloctort
    help, xloctort
    cghistoplot, avgintesity, binsize=flmtintensitybinsize, /noerase, color='red',position=[0.75,0.42,0.97,0.66], title='Fil-Avg SMLM Intensity',mininput = 0

    t3a = 'Mean Length.:' +string(mean(contourlength*pixelsize,/nan), format='(F8.4)')
    t3b = 'St.dev. Length.:' +string(stddev(contourlength*pixelsize), format='(F8.4)')

    t4a = 'Mean Tort.:' +string(mean(tortuosity,/nan), format='(F8.4)')
    t4b = 'St.dev. Tort.:' +string(min(tortuosity), format='(F8.4)')

    t4c = 'Mean Intensity:' +string(mean(avgintesity,/nan), format='(F6.2)')
    t4d = 'Min Intensity:' +string(min(avgintesity), format='(F6.2)')
    t4e = 'Max Intensity:' +string(max(avgintesity), format='(F6.2)')

    cgtext, 0.35,0.01,t3a, color='blue',/normal
    cgtext, 0.35,0.0235,t3b, color='blue',/normal

    cgtext, 0.70,0.01,t4a, color='blue',/normal
    cgtext, 0.70,0.0235,t4b, color='blue',/normal

    cgtext, 0.8,0.6,t4c, color='blue',/normal
    cgtext, 0.8,0.6-0.0145,t4e, color='blue',/normal
    cgtext, 0.8,0.6-0.0145*2,t4d, color='blue',/normal

    porearea = pore_table[3,*]
    cghistoplot, porearea[okpore]*pixelsize*pixelsize, binsize=poreareabinsize, mininput=0, /noerase, color='red',position=[0.075,0.42,0.30,0.66], title='Pore Area',xtitle='Micron^2',maxinput=1.0, $
      loc=x_pore, histdata=y_pore
    cghistoplot, porearea[okpore]*pixelsize*pixelsize, binsize=poreareabinsize, mininput = 0,/noerase, color='red',position=[0.42,0.42,0.64,0.66], title='SemiLog:Pore Area',xtitle='Micron^2',maxinput=1.0,/log,ytickformat='(E10.2)'

    t5e = 'MedArea:' +string(median(porearea[okpore]*pixelsize*pixelsize), format='(F10.4)')+' um^2'
    t5a = 'AvgArea:' +string(mean(porearea[okpore]*pixelsize*pixelsize,/nan), format='(F10.4)')+' um^2'
    t5b = 'StdArea:' +string(stddev(porearea[okpore]*pixelsize*pixelsize), format='(F10.4)')+' um^2'
    t5c = 'AvgDia:'+string(2*sqrt(mean(porearea[okpore]*pixelsize*pixelsize,/nan)/!PI), format='(F10.4)')+' um'
    t5d = 'TotArea:'+string(total(porearea[okpore]*pixelsize*pixelsize,/nan), format='(F10.4)')+' um^2'
    t5f = 'MedDia:'+string(2*sqrt(median(porearea[okpore]*pixelsize*pixelsize)/!PI), format='(F10.4)')+' um'
    cgtext, 0.10,0.6+0.0145,t5e, color='blue',/normal
    cgtext, 0.10,0.6,t5a, color='blue',/normal
    cgtext, 0.10,0.6-.0145,t5b, color='blue',/normal
    cgtext, 0.10,0.6-.0145*2,t5c, color='blue',/normal
    cgtext, 0.10,0.6-.0145*3,t5f, color='blue',/normal
    cgtext, 0.10,0.6-.0145*4,t5d, color='blue',/normal

    cghistoplot, angle_table[6,*], binsize=anglebinsize, /noerase, color='red',position=[0.08,0.73,0.55,0.95], title='Interfilament Angle (optim) (degree)',mininput=0,xtitle='deg.', $
      loc=x_angle, histdata=y_angle
    t6a = 'AvgAngle:' +string(mean(angle_table[6,*],/nan), format='(F10.4)')+' deg'
    t6b = 'StdAngle:' +string(stddev(angle_table[6,*]), format='(F10.4)')+' deg'

    index = where(angle_table[6,*] gt 180, largeanagle)
    ind90 = where(angle_table[6,*] eq 90, n90)
    ind180 = where(angle_table[6,*] eq 180, n180)
    t6c = 'Angle>180:'+string(largeanagle*100./n_elements(aid), format='(F10.4)')+' %'
    t6d = 'Angle=90 or 180:'+string((n90+n180)*100./n_elements(aid), format='(F10.4)')+' %'
    t6e = 'FilamentDensity:'+string(total(contourlength)/(total(porearea[okpore])*pixelsize), format='(F12.7)')+' um-1'
    print, t6d

    cgtext, 0.58,0.95-0.015*4, t6a,color='blue',/normal
    cgtext, 0.58,0.95-0.015*5, t6b,color='blue',/normal
    cgtext, 0.58,0.95-0.015*6, t6c,color='blue',/normal
    cgtext, 0.58,0.95-0.015*8, t6e,color='blue',/normal

    if keyword_set(export) and keyword_set(cell) then begin

      fn = Dialog_Pickfile(/write,get_path=fpath,filter=['*histo.csv'],title='Export pooled histogram data into *.csv file')
      if fn eq '' then return
      cd,fpath
      fn_angle=meshwork_AddExtension(fn,'_angle_histo.csv')
      fn_pore=meshwork_AddExtension(fn,'_pore_histo.csv')
      fn_fil=meshwork_AddExtension(fn,'_filseglength_histo.csv')

      table_header = keyword_set(noheader)?!null: [properties.datafile, t1a,t1b,t2a,t2b,t2c,t3a,t3b,t4a,t4b,t4c,t4d,t4e,t5a,t5b,t5c,t5d,t5e,t5f,t6a,t6b,t6c,t6d,t6e]
      write_csv, fn_angle,x_angle ,y_angle , double(y_angle)/max(y_angle),table_header=table_header,header=keyword_set(noheader)?!null:['Angle(deg)','#','Norm']
      meshwork_io_status,event,text='Export:'+fn_angle
      print,'Export:'+fn_angle
      write_csv, fn_pore,x_pore ,y_pore ,double(y_pore)/max(y_pore),table_header=table_header,header=keyword_set(noheader)?!null:['Area(um^2)','#','Norm']
      meshwork_io_status,event,text='Export:'+fn_pore
      print,'Export:'+fn_pore
      write_csv, fn_fil,x_fil ,y_fil ,double(y_fil)/max(y_fil),table_header=table_header,header=keyword_set(noheader)?!null:['Length(um)','#','Norm']
      meshwork_io_status,event,text='Export:'+fn_fil
      print,'Export:'+fn_fil
      z=dialog_message('Finish exporting..')
      return
    endif



    ;      numfilpernode = total(node_table[5:12,*] ne 0, 1)
    ;      help, numfilpernode


    return
  endif

  if keyword_set(batch) then begin
    filename = Dialog_Pickfile(/read,/directory,get_path=fpath,title='Select a directory of _meshworks.sav files')
    if filename eq '' then return
    cd, fpath
    result = file_search(filename,'*_meshworks.sav',/fold_case)
    numfiles = n_elements(result)
    if n_elements(result) lt 1 then return

    print, 'Found..', numfiles, ' files:'
    print, result

    if keyword_set(export) then begin
      if keyword_set(pooled) then fn = Dialog_Pickfile(/write,get_path=fpath,filter=['*pooled.csv'],title='Export pooled data points into *.csv file') $
      else   fn = Dialog_Pickfile(/write,get_path=fpath,filter=['*pooledhisto.csv'],title='Export pooled histograms into *.csv file')
      if fn eq '' then return
    endif

    filamentdensity = dblarr(numfiles)
    angledatapts = ptrarr(numfiles)
    lengthdatapts = ptrarr(numfiles)
    poresizedatapts = ptrarr(numfiles)

    anglehistoptrarr = ptrarr(numfiles)
    poreareaptrarr = ptrarr(numfiles)
    fillengthptrarr = ptrarr(numfiles)
    filtortptrarr = ptrarr(numfiles)
    filintensptrarr = ptrarr(numfiles)
    nodeintensptrarr = ptrarr(numfiles)
    nodecodeptrarr = ptrarr(numfiles)

    for i =0, numfiles-1 do begin
      print, 'Loading files...:',result[i]
      meshwork_io_status,event,text = 'Loading files...:'+result[i]
      meshwork_io,event,/loadsav,batch=result[i]
      meshwork_display,event,/reset
      meshwork_display_refresh, event

      widget_control, widget_info(event.top,find_by_uname='WID_TABLE_PORES'), get_value=pore_table
      widget_control, widget_info(event.top,find_by_uname='WID_TABLE_NODES'), get_value=node_table
      widget_control, widget_info(event.top,find_by_uname='WID_TABLE_FILAMENTS'), get_value=filament_table
      widget_control, widget_info(event.top,find_by_uname='WID_TABLE_ANGLES'), get_value=angle_table

      nid = node_table[0,*]
      fid = filament_table[0,*]
      pid = pore_Table[0,*]
      aid = angle_table[0,*]
      pcode = pore_table[5,*]
      okpore = where(pcode ge 0, countgoodpore)

      t1a = 'Total # nodes:'+ string(n_elements(nid), format = '(I10)')+ ' Total # filaments:'+ string(n_elements(fid),format ='(I10)')
      t1b = 'Total # pores:'+string(n_elements(pid), format='(I10)')+' Total # of angles:'+string(n_elements(aid), format='(I10)')
      meshwork_io_status,event,text=t1

      if ~keyword_set(noerase) then cgerase,'white' ;image,bytscl(bytarr(1024,1024)+1)
      cgtext, 0.025,0.98, properties.datafile,color='blue',/normal,charsize =1.5
      cgtext, 0.58,0.95, t1a,color='blue',/normal
      cgtext, 0.58,0.95-0.015, t1b,color='blue',/normal

      nodeintensity = reform(node_table[3,*])
      cghistoplot, nodeintensity, binsize=nodeintensitybinsize, /noerase, color='red',position=[0.075,0.08,0.30,0.34], title='SMLM intensity at nodes (avg)',mininput=0,loc=xlocnode,histdata=histdata

      nodeintensptrarr[i] = ptr_new({x:xlocnode,y:histdata})
      nodecodeptrarr[i] = ptr_new(total(node_table[5:12,*] ne 0, 1))

      t2a = 'Mean Intensity:' +string(mean(nodeintensity,/nan), format='(F6.2)')
      t2b = 'Min Intensity:' +string(min(nodeintensity), format='(F6.2)')
      t2c = 'Max Intensity:' +string(max(nodeintensity), format='(F6.2)')
      cgtext, 0.01,0.01,t2b,color='blue',/normal
      cgtext, 0.01,0.0235,t2c, color='blue',/normal
      cgtext, 0.01,0.0235+0.0135,t2a, color='blue',/normal

      contourlength = filament_table[8,*]
      vectorlength = sqrt((filament_table[4,*]-filament_table[6,*])^2+ (filament_table[5,*]-filament_table[7,*])^2)
      tortuosity = (contourlength+2)/vectorlength
      avgintesity = filament_table[13,*]
      cghistoplot, contourlength*pixelsize, binsize=flmntlengthbinsize, /noerase, color='red',position=[0.42,0.08,0.64,0.34], title='Fil-Seg contour-length (micron)',xtitle = 'micron', mininput=0,histdata=histdata,loc=xloc
      fillengthptrarr[i] = ptr_new({x:xloc ,y:histdata })
      cghistoplot, tortuosity, binsize=flmnttortbinsize, /noerase, color='red',position=[0.75,0.08,0.97,0.34], title='Fil-Seg Tortuosity',mininput=1, loc=xloc ,histdata=histdata
      filtortptrarr[i] = ptr_new({x:xloc ,y:histdata })
      cghistoplot, avgintesity, binsize=flmtintensitybinsize, /noerase, color='red',position=[0.75,0.42,0.97,0.66], title='Fil-Avg SMLM Intensity',mininput = 0,histdata=histdata, loc=xloc
      filintensptrarr[i] = ptr_new({x:xloc ,y:histdata })

      t3a = 'Mean Length.:' +string(mean(contourlength*pixelsize,/nan), format='(F8.4)')
      t3b = 'St.dev. Length.:' +string(stddev(contourlength*pixelsize), format='(F8.4)')

      t4a = 'Mean Tort.:' +string(mean(tortuosity,/nan), format='(F8.4)')
      t4b = 'St.dev. Tort.:' +string(min(tortuosity), format='(F8.4)')

      t4c = 'Mean Intensity:' +string(mean(avgintesity,/nan), format='(F6.2)')
      t4d = 'Min Intensity:' +string(min(avgintesity), format='(F6.2)')
      t4e = 'Max Intensity:' +string(max(avgintesity), format='(F6.2)')

      cgtext, 0.35,0.01,t3a, color='blue',/normal
      cgtext, 0.35,0.0235,t3b, color='blue',/normal

      cgtext, 0.70,0.01,t4a, color='blue',/normal
      cgtext, 0.70,0.0235,t4b, color='blue',/normal

      cgtext, 0.8,0.6,t4c, color='blue',/normal
      cgtext, 0.8,0.6-0.0145,t4e, color='blue',/normal
      cgtext, 0.8,0.6-0.0145*2,t4d, color='blue',/normal

      porearea = pore_table[3,*]
      cghistoplot, porearea[okpore]*pixelsize*pixelsize, binsize=poreareabinsize, mininput=0, /noerase, color='red',position=[0.075,0.42,0.30,0.66], title='Pore Area',xtitle='Micron^2',maxinput=1.0, histdata=histdata,loc=xloc
      cghistoplot, porearea[okpore]*pixelsize*pixelsize, binsize=poreareabinsize, mininput = 0,/noerase, color='red',position=[0.42,0.42,0.64,0.66], title='SemiLog:Pore Area',xtitle='Micron^2',maxinput=1.0,/log,ytickformat='(E10.2)'
      poreareaptrarr[i] = ptr_new({x:xloc ,y:histdata })

      t5e = 'MedArea:' +string(median(porearea[okpore]*pixelsize*pixelsize), format='(F10.4)')+' um^2'
      t5a = 'AvgArea:' +string(mean(porearea[okpore]*pixelsize*pixelsize,/nan), format='(F10.4)')+' um^2'
      t5b = 'StdArea:' +string(stddev(porearea[okpore]*pixelsize*pixelsize), format='(F10.4)')+' um^2'
      t5c = 'AvgDia:'+string(2*sqrt(mean(porearea[okpore]*pixelsize*pixelsize,/nan)/!PI), format='(F10.4)')+' um'
      t5d = 'TotArea:'+string(total(porearea[okpore]*pixelsize*pixelsize,/nan), format='(F10.4)')+' um^2'
      t5f = 'MedDia:'+string(2*sqrt(median(porearea[okpore]*pixelsize*pixelsize)/!PI), format='(F10.4)')+' um'
      cgtext, 0.10,0.6+0.0145,t5e, color='blue',/normal
      cgtext, 0.10,0.6,t5a, color='blue',/normal
      cgtext, 0.10,0.6-.0145,t5b, color='blue',/normal
      cgtext, 0.10,0.6-.0145*2,t5c, color='blue',/normal
      cgtext, 0.10,0.6-.0145*3,t5f, color='blue',/normal
      cgtext, 0.10,0.6-.0145*4,t5d, color='blue',/normal

      cghistoplot, angle_table[6,*], binsize=anglebinsize, /noerase, color='red',position=[0.08,0.73,0.55,0.95], title='Interfilament Angle (optim) (degree)',mininput=0,xtitle='deg.',histdata=histdata,loc=xloc
      anglehistoptrarr[i] = ptr_new({x:xloc ,y:histdata })
      t6a = 'AvgAngle:' +string(mean(angle_table[6,*],/nan), format='(F10.4)')+' deg'
      t6b = 'StdAngle:' +string(stddev(angle_table[6,*]), format='(F10.4)')+' deg'

      index = where(angle_table[6,*] gt 180, largeanagle)
      ind90 = where(angle_table[6,*] eq 90, n90)
      ind180 = where(angle_table[6,*] eq 180, n180)
      t6c = 'Angle>180:'+string(largeanagle*100./n_elements(aid), format='(F10.4)')+' %'
      t6d = 'Angle=90 or 180:'+string((n90+n180)*100./n_elements(aid), format='(F10.4)')+' %'
      t6e = 'FilamentDensity:'+string(total(contourlength)/(total(porearea[okpore])*pixelsize), format='(F12.7)')+' um-1'
      print, t6d
      filamentdensity[i] = total(contourlength)/(total(porearea[okpore])*pixelsize)

      cgtext, 0.58,0.95-0.015*4, t6a,color='blue',/normal
      cgtext, 0.58,0.95-0.015*5, t6b,color='blue',/normal
      cgtext, 0.58,0.95-0.015*6, t6c,color='blue',/normal
      cgtext, 0.58,0.95-0.015*8, t6e,color='blue',/normal

      angledatapts[i] = ptr_new(angle_table[6,*])
      lengthdatapts[i] = ptr_new(contourlength*pixelsize)
      poresizedatapts[i] = ptr_new(porearea[okpore]*pixelsize*pixelsize)
    endfor

    ;    help, anglehistoptrarr ,    poreareaptrarr  ,    fillengthptrarr ,    filtortptrarr ,    filintensptrarr ,    nodeintensptrarr
    ;    nodeintensitybinsize = 20.
    ;    flmtintensitybinsize = 20.
    ;    poreareabinsize =  0.01
    ;    anglebinsize = 5.
    ;    flmntlengthbinsize = 0.05
    ;    flmnttortbinsize = 0.1

    numentry = 0
    for i = 0, numfiles-1 do begin
      thisstruct = *nodeintensptrarr[i]
      numentry  = max([numentry, n_elements(thisstruct.x)])
    endfor
    x_nodeintensity = findgen(numentry)*nodeintensitybinsize+0.5*nodeintensitybinsize
    nodeintensity = dblarr(numentry, numfiles)
    for i = 0, numfiles-1 do begin
      thisy = (*nodeintensptrarr[i]).y
      nodeintensity[0:n_elements(thisy)-1,i]=double(thisy)/total(thisy)
    endfor
    cgplot, x_nodeintensity, numfiles, /nodata, color='red',position=[0.075,0.08,0.30,0.34], title='SMLM intensity at nodes (avg)',ytitle='dataset #'
    cgimage, nodeintensity,/noerase,/scale, position=[0.075,0.08,0.30,0.34], ctindex =3

    ;==
    numentry = 0
    for i = 0, numfiles-1 do begin
      thisstruct = *fillengthptrarr[i]
      numentry  = max([numentry, n_elements(thisstruct.x)])
    endfor
    x_flmntlength = findgen(numentry)*flmntlengthbinsize+0.5*flmntlengthbinsize
    flmntlength = dblarr(numentry, numfiles)
    for i = 0, numfiles-1 do begin
      thisy = (*fillengthptrarr[i]).y
      flmntlength[0:n_elements(thisy)-1,i]=double(thisy)/total(thisy)
    endfor
    cgplot, x_flmntlength, numfiles, /nodata,/noerase, color='red',position=[0.42,0.08,0.64,0.34], title='Fil-Seg contour-length (micron)',xtitle = 'micron', ytitle='dataset #'
    cgimage, flmntlength,/noerase,/scale, position=[0.42,0.08,0.64,0.34], ctindex =3

    ;==
    numentry = 0
    for i = 0, numfiles-1 do begin
      thisstruct = *filtortptrarr[i]
      numentry  = max([numentry, n_elements(thisstruct.x)])
    endfor
    x_flmnttort = findgen(numentry)*flmnttortbinsize+0.5*flmnttortbinsize
    flmnttort = dblarr(numentry, numfiles)
    for i = 0, numfiles-1 do begin
      thisy = (*filtortptrarr[i]).y
      flmnttort[0:n_elements(thisy)-1,i]=double(thisy)/total(thisy)
    endfor
    cgplot, x_flmnttort, numfiles, /nodata,/noerase, color='red',position=[0.75,0.08,0.97,0.34], title='Fil-Seg Tortuosity',xtitle = 'a.u.', ytitle='dataset #'
    cgimage, flmnttort,/noerase,/scale, position=[0.75,0.08,0.97,0.34], ctindex =3


    ;==
    numentry = 0
    for i = 0, numfiles-1 do begin
      thisstruct = *filintensptrarr[i]
      numentry  = max([numentry, n_elements(thisstruct.x)])
    endfor
    x_flmntintensity = findgen(numentry)*flmtintensitybinsize+0.5*flmtintensitybinsize
    flmntintensity = dblarr(numentry, numfiles)
    for i = 0, numfiles-1 do begin
      thisy = (*filintensptrarr[i]).y
      flmntintensity[0:n_elements(thisy)-1,i]=double(thisy)/total(thisy)
    endfor
    cgplot, x_flmntintensity, numfiles, /nodata,/noerase, color='red',position=[0.75,0.42,0.97,0.66], title='Fil-Avg SMLM Intensity',xtitle = 'a.u.', ytitle='dataset #'
    cgimage, flmntintensity,/noerase,/scale, position=[0.75,0.42,0.97,0.66], ctindex =3

    ;==
    numentry = 0
    for i = 0, numfiles-1 do begin
      thisstruct = *poreareaptrarr[i]
      numentry  = max([numentry, n_elements(thisstruct.x)])
    endfor
    x_porearea = findgen(numentry)*poreareabinsize+0.5*poreareabinsize
    porearea = dblarr(numentry, numfiles)
    for i = 0, numfiles-1 do begin
      thisy = (*poreareaptrarr[i]).y
      porearea[0:n_elements(thisy)-1,i]=double(thisy)/total(thisy)
    endfor
    cgplot, x_porearea, numfiles, /nodata,/noerase, color='red',position=[0.075,0.42,0.30,0.66], title='Pore Area (Log10 density)',xtitle = 'um^2', ytitle='dataset #'
    cgimage, alog10(porearea),/noerase,/scale, position=[0.075,0.42,0.30,0.66], ctindex =3

    ;===
    numentry = 0
    for i = 0, numfiles-1 do begin
      thisstruct = *anglehistoptrarr[i]
      numentry  = max([numentry, n_elements(thisstruct.x)])
    endfor
    x_angle = findgen(numentry)*anglebinsize+0.5*anglebinsize
    angle = dblarr(numentry, numfiles)
    for i = 0, numfiles-1 do begin
      thisy = (*anglehistoptrarr[i]).y
      angle[0:n_elements(thisy)-1,i]=double(thisy)/total(thisy)
    endfor
    cgplot, x_angle, numfiles, /nodata,/noerase, color='red',position=[0.42,0.42,0.64,0.66], title='Interfilament Angle (optim) (degree)',xtitle = 'a.u.', ytitle='dataset #'
    cgimage, angle,/noerase,/scale, position=[0.42,0.42,0.64,0.66], ctindex =3

    ;==
    erry = stddev(angle, dimension=2)
    cgplot, x_angle, mean(angle, dimension=2), position = [0.42,0.73,0.64,0.95],/noerase, title='Interfilament Angle (optim) (degree)',xtitle = 'degree',color='blue',thick=2,ytitle='Norm Freq',$
      err_yhigh=erry, err_ylow=erry,psym=10
    for i = 0, numfiles-1 do cgplots,x_angle,angle[*,i],psym=3,color='red',symsize=2

    erry = stddev(flmntlength, dimension=2)
    cgplot, x_flmntlength, mean(flmntlength, dimension=2), position = [0.75,0.73,0.97,0.95],/noerase, title='Fil-Seg contour-length (um)',xtitle = 'um',color='blue',thick=2,ytitle='Norm Freq',$
      err_yhigh=erry, err_ylow=erry,psym=10
    for i = 0, numfiles-1 do cgplots,x_flmntlength,flmntlength[*,i],psym=3,color='red',symsize=2

    erry = stddev(porearea, dimension=2)
    cgplot, x_porearea, mean(porearea, dimension=2), position = [0.075,0.73,0.30,0.95],/noerase, title='Pore Area',xtitle = 'um^2',color='blue',thick=2,ytitle='Norm Freq',err_yhigh=erry, err_ylow=erry,psym=10
    for i = 0, numfiles-1 do cgplots,x_porearea,porearea[*,i],psym=3,color='red',symsize=2

    totangle =[]
    totpore = []
    totfil = []
    totncode = []
    for i=0, numfiles-1 do begin
      totangle = [totangle, reform(*angledatapts[i])]
      totpore = [totpore, reform(*poresizedatapts[i])]
      totfil = [totfil, reform(*lengthdatapts[i])]
      totncode = [totncode, reform(*nodecodeptrarr[i])]
    endfor
    numangle  =n_elements(totangle)
    numpore = n_elements(totpore)
    numfil = n_elements(totfil)

    b1a = 'MedArea:' +string(median(totpore), format='(F10.4)')+' um^2'
    b1b = 'AvgArea:' +string(mean(totpore,/nan), format='(F10.4)')+' um^2'
    b1c = 'StdArea:' +string(stddev(totpore), format='(F10.4)')+' um^2'
    b1d = 'AvgDia:'+string(2*sqrt(mean(totpore,/nan)/!PI), format='(F10.4)')+' um'
    b1f = 'TotArea:'+string(total(totpore,/nan), format='(F10.4)')+' um^2'
    b1e = 'MedDia:'+string(2*sqrt(median(totpore)/!PI), format='(F10.4)')+' um'

    b0 = '# filaments:'+string(numfil, format='(I10)')+'  #pores:'+string(numpore, format='(I10)')+'  #angles:'+string(numangle, format='(I10)')+' #files:'+string(numfiles,format='(I5)')
    cgtext, 0.025,0.98,fpath,color='blue',/normal,charsize=1.5
    cgtext,0.025 ,0.025, b0,color='blue',/normal

    cgtext, 0.10,0.92,b1a, color='blue',/normal
    cgtext, 0.10,0.92-0.0145*1,b1b, color='blue',/normal
    cgtext, 0.10,0.92-0.0145*2,b1c, color='blue',/normal
    cgtext, 0.10,0.92-0.0145*3,b1d, color='blue',/normal
    cgtext, 0.10,0.92-0.0145*4,b1e, color='blue',/normal
    cgtext, 0.10,0.92-0.0145*5,b1f, color='blue',/normal

    b2a = 'MedLength:' +string(median(totfil), format='(F10.4)')+' um'
    b2b = 'AvgLength:' +string(mean(totfil,/nan), format='(F10.4)')+' um'
    b2c = 'StdLength:' +string(stddev(totfil), format='(F10.4)')+' um'
    b2d = 'FilDensity:'+string(total(totfil)/total(totpore), format='(F12.7)')+' um-1'
    cgtext, 0.8,0.92,b2a, color='blue',/normal
    cgtext, 0.8,0.92-0.0145*1,b2b, color='blue',/normal
    cgtext, 0.8,0.92-0.0145*2,b2c, color='blue',/normal
    cgtext, 0.8,0.92-0.0145*3,b2d, color='blue',/normal

    b3a = 'MedAngle:' +string(median(abs(totangle)), format='(F10.4)')+' deg'
    b3b = 'AvgAngle:' +string(mean(totangle,/nan), format='(F10.4)')+' deg'
    b3c = 'StdAngle:' +string(stddev(totangle), format='(F10.4)')+' deg'

    cgtext, 0.52,0.92,b3a, color='blue',/normal
    cgtext, 0.52,0.92-0.0145*1,b3b, color='blue',/normal
    cgtext, 0.52,0.92-0.0145*2,b3c, color='blue',/normal

    ncodenormal = total(totncode eq 1)
    ncode2 = total(totncode eq 2)
    ncode3 = total(totncode eq 3)
    ncode4 = total(totncode eq 4)
    ncode5 = total(totncode eq 5)
    ncodebig = total(totncode gt 5)

    b4x =  'Node-code normal ='+string(ncodenormal, format = '(I10)')+ $
      'Node-code (2) ='+string(ncode2, format = '(I10)') + $
      'Node-code (3) ='+string(ncode3, format = '(I10)') + $
      'Node-code (4) ='+string(ncode4, format = '(I10)') + $
      'Node-code (5) ='+string(ncode5, format = '(I10)')+$
      'Node-code >5 ='+string(ncodebig, format = '(I10)')
    print, b4x

    if keyword_set(export) then begin
      table_header = keyword_set(noheader)?!null:[filename,b0,b1a,b1b,b1c,b1d,b1e,b1f,b2a,b2b,b2c,b2d,b3a,b3b,b3c,b4x]
      cd,fpath

      if keyword_set(pooled) then begin
        fn_angle=meshwork_AddExtension(fn,'_angle_pooled.csv')
        fn_pore=meshwork_AddExtension(fn,'_pore_pooled.csv')
        fn_fil=meshwork_AddExtension(fn,'_filseglength_pooled.csv')

        write_csv, fn_angle,totangle , table_header=table_header,header=keyword_set(noheader)?!null:['Angle(deg)']
        meshwork_io_status,event,text='Export:'+fn_angle
        print,'Export:'+fn_angle
        write_csv, fn_pore,totpore ,table_header=table_header,header=keyword_set(noheader)?!null:['Area(um^2)']
        meshwork_io_status,event,text='Export:'+fn_pore
        print,'Export:'+fn_pore
        write_csv, fn_fil,totfil ,table_header=table_header,header=keyword_set(noheader)?!null:['Length(um)']
        meshwork_io_status,event,text='Export:'+fn_fil
        print,'Export:'+fn_fil

        z=dialog_message('Finish exporting pooled data..')

        return
      endif else begin
        fn_angle=meshwork_AddExtension(fn,'_angle_pooledhisto.csv')
        fn_pore=meshwork_AddExtension(fn,'_pore_pooledhisto.csv')
        fn_fil=meshwork_AddExtension(fn,'_filseglength_pooledhisto.csv')
        fn_fildensity = meshwork_AddExtension(fn,'_fildensity.csv')

        angleoutput = dblarr(3+numfiles,n_elements(x_angle))
        angleoutput[0,*]=x_angle
        angleoutput[1,*]=mean(angle, dimension=2)
        angleoutput[2,*]=stddev(angle, dimension=2)
        angleoutput[3:-1,*]=transpose(angle)
        write_csv, fn_angle, angleoutput, table_header=table_header,header=keyword_set(noheader)?!null:['Angle(deg)','mean','std',result]
        meshwork_io_status,event,text='Export:'+fn_angle
        print,'Export:'+fn_angle

        poreoutput = dblarr(3+numfiles,n_elements(x_porearea))
        poreoutput[0,*]=x_porearea
        poreoutput[1,*]=mean(porearea, dimension=2)
        poreoutput[2,*]=stddev(porearea, dimension=2)
        poreoutput[3:-1,*]=transpose(porearea)
        write_csv, fn_pore, poreoutput, table_header=table_header,header=keyword_set(noheader)?!null:['Area(um^2)','mean','std',result]
        meshwork_io_status,event,text='Export:'+fn_pore
        print,'Export:'+fn_pore

        lengthoutput = dblarr(3+numfiles,n_elements(x_flmntlength))
        lengthoutput[0,*]=x_flmntlength
        lengthoutput[1,*]=mean(flmntlength, dimension=2)
        lengthoutput[2,*]=stddev(flmntlength, dimension=2)
        lengthoutput[3:-1,*]=transpose(flmntlength)
        write_csv, fn_fil, lengthoutput, table_header=table_header,header=keyword_set(noheader)?!null:['Length(um)','mean','std',result]
        meshwork_io_status,event,text='Export:'+fn_fil
        print,'Export:'+fn_fil

        write_csv, fn_fildensity, filamentdensity ,table_header=['Filament density:',filename]
        meshwork_io_status,event,text='Export:'+fn_fildensity
        print,'Export:'+fn_fildensity
        z=dialog_message('Finish exporting pooled histograms..')

      endelse
      return
    endif
    return
  endif


end

pro meshwork_table, event, ntable=ntable, ptable=ptable, ftable=ftable, atable=atable, selection=selection, export=export, statistics=statistics, noerase=noerase, $
  astertable=astertable

  Common meshwork_dataset, properties, image_set, filamentmask, nodemask, node_table, pore_table, filament_table, angle_table, aster_table

  if ~keyword_set(noerase) then cgerase, 'white';image,bytscl(bytarr(1024,1024)+1)
  if keyword_set(ntable) and size(node_table,/type) ne 0 then begin
    thistable = node_table
    table_header=[ 'Node-ID', 'X-center','Y-center', 'SMLM intensity','N-code','F-ID1','F-ID2','F-ID3','F-ID4','F-ID5','F-ID6','F-ID7','F-ID8', $
      'P-ID1','P-ID2','P-ID3','P-ID4','P-ID5','P-ID6','P-ID7','P-ID8']
  endif
  if keyword_set(ptable) and size(pore_table,/type) ne 0 then begin
    thistable = pore_table
    table_header=[ 'pore-ID', 'X-center','Y-center','Area','Vertices','P-Code']
  endif
  if keyword_set(ftable) and size(filament_table,/type) ne 0 then begin
    thistable = filament_table
    table_header=[ 'filament-ID', 'Node-I','Node-F','F-code','X-i','Y-i','X-f','Y-f','Contour-length','Angle-I','Angle-F','GrAng-I','GrAng-F','Avg-SMLM-intensity']
  endif
  if keyword_set(atable) and size(angle_table,/type) ne 0 then begin
    thistable = filament_table
    table_header=['Angle-ID', 'Node-ID','X-center','Y-center', 'Interfilament-Angle','Fil-ID1','Fil-ID2','Fil-1-length','Fil-2-length','Angle-Code','P-smlm-avg','P-smlm-max','P-edge-avg']
  endif
  widget_control,widget_info(event.top,find_by_uname='WID_TABLE_ASTERS'),get_value=aster_table
  if keyword_set(astertable) and size(aster_table,/type) ne 0 then begin
    thistable = aster_table
    table_header=[ 'Aster-ID', 'X-center','Y-center', 'Area','AvgIntens','PeakIntens','Coord-#','aster-code']
  endif

  if keyword_set(export) and keyword_set(astertable) then begin
    filename = Dialog_Pickfile(/write,get_path=fpath,filter=['*.csv','*.txt'],title='Save aster table to comma-delimited file')
    if filename eq '' then return
    cd,fpath
    filename  =meshwork_addextension(filename,'.csv')
    widget_control,widget_info(event.top, find_by_uname='WID_TABLE_ASTERS'), get_value=aster_table
    write_csv,filename,aster_table,header=table_header
    dummy = dialog_message('Saving completed...')
    return
  endif

  if keyword_set(export) and keyword_set(atable) then begin
    filename = Dialog_Pickfile(/write,get_path=fpath,filter=['*.csv','*.txt'],title='Save angle table to comma-delimited file')
    if filename eq '' then return
    cd,fpath
    filename  =meshwork_addextension(filename,'.csv')
    widget_control,widget_info(event.top, find_by_uname='WID_TABLE_ANGLES'), get_value=angle_table
    write_csv,filename,angle_table,header=table_header
    dummy = dialog_message('Saving completed...')
    return
  endif

  if keyword_set(export) and keyword_set(ftable) then begin
    filename = Dialog_Pickfile(/write,get_path=fpath,filter=['*.csv','*.txt'],title='Save filament table to comma-delimited file')
    if filename eq '' then return
    cd,fpath
    filename  =meshwork_addextension(filename,'.csv')
    widget_control,widget_info(event.top, find_by_uname='WID_TABLE_FILAMENTS'), get_value=angle_table
    write_csv,filename,filament_table,header=table_header
    dummy = dialog_message('Saving completed...')
    return
  endif

  if keyword_set(export) and keyword_set(ntable) then begin
    filename = Dialog_Pickfile(/write,get_path=fpath,filter=['*.csv','*.txt'],title='Save node table to comma-delimited file')
    if filename eq '' then return
    cd,fpath
    filename  =meshwork_addextension(filename,'.csv')
    widget_control,widget_info(event.top, find_by_uname='WID_TABLE_NODES'), get_value=node_table
    write_csv,filename,node_table,header=table_header
    dummy = dialog_message('Saving completed...')
    return
  endif

  if keyword_set(export) and keyword_set(ptable) then begin
    filename = Dialog_Pickfile(/write,get_path=fpath,filter=['*.csv','*.txt'],title='Save pore table to comma-delimited file')
    if filename eq '' then return
    cd,fpath
    filename  =meshwork_addextension(filename,'.csv')
    widget_control,widget_info(event.top, find_by_uname='WID_TABLE_PORES'), get_value=pore_table
    write_csv,filename,pore_table,table_header=table_header
    dummy = dialog_message('Saving completed...')
    return
  endif

  if keyword_set(statistics) and keyword_set(astertable) then begin
    widget_control,widget_info(event.top, find_by_uname='WID_TABLE_ASTERS'), get_value=aster_table
    print, 'Asters analysis'

    cgplot, aster_table[3,*],aster_table[4,*], psym=2, /noerase, color='red',symsize=1, position=[0.075,0.08,0.30,0.30],xtitle='Area(pix)',ytitle='Avg Intensity',title='Area vs. Avg.Intensity'
    cghistoplot, aster_table[6,*], nbins=10, /noerase, color='red',position=[0.37,0.07,0.55,0.31], title='Coordination # '
    cgplot, aster_table[6,*],aster_table[3,*], psym=2, /noerase, color='red',symsize=2, position=[0.65,0.08,0.96,0.30],xtitle='Coordination #',ytitle='Area'
    meancoord = mean(aster_table[6,*],/nan)
    mediancoord = median(aster_table[6,*])
    stdcoord = stddev(aster_table[6,*],/nan)
    t1 = 'Total Asters:'+string(n_elements(aster_table[0,*]), format='(I10)')+' Mean Coordiation #:'+string(meancoord,format = '(F10.2)') + ' Median Coordiation #:'+string(mediancoord,format = '(F10.2)')+ ' St.dev Coordiation #:'+string(stdcoord,format = '(F10.2)')
    cgtext, 0.025,0.95,t1,/normal,color='red'
    return
  endif

  if keyword_set(statistics) and keyword_set(atable) then begin
    widget_control,widget_info(event.top, find_by_uname='WID_TABLE_ANGLES'), get_value=angle_table
    print, 'Angle analysis'
    interfangle = reform(angle_table[4,*])
    cghistoplot, angle_table[4,*], nbins=60, /noerase, color='red',position=[0.37,0.07,0.55,0.31], title='Interfilament Angle (fit) (degree)'
    cghistoplot, angle_table[5,*], nbins=60, /noerase, color='red',position=[0.37,0.39,0.55,0.64], title='Interfilament Angle (direct)  (degree)'
    cghistoplot, angle_table[6,*], nbins=60, /noerase, color='red',position=[0.37,0.72,0.55,0.94], title='Interfilament Angle (optim) (degree)'
    cgplot, angle_table[9,*],angle_table[10,*], psym=3, /noerase, color='red',symsize=2, position=[0.075,0.08,0.30,0.30],xtitle='Fil-1-length',ytitle='Fil-2-length',/isotropic,/nodata
    f1f2 = hist_2d(angle_table[9,*],bin1=1,min1=0,angle_table[10,*],bin2=1,min2=0)
    cgimage, f1f2,position=[0.075,0.08,0.30,0.30],/noerase, ctindex=2,/scale
    cgplot, angle_table[9,*],angle_table[4,*], psym=3, /noerase, color='red',symsize=2, position=[0.65,0.08,0.96,0.30],xtitle='Fil-max-length',ytitle='fit angle(deg)',isotropic=0,/nodata
    a1length = hist_2d(max(angle_table[9:10,*], dimension = 1),bin1=1,min1=0,angle_table[4,*],bin2=1,min2=0)
    cgimage, a1length,position=[0.65,0.08,0.96,0.30],/noerase, ctindex=2,/scale
    cgplot, angle_table[10,*],angle_table[4,*], psym=3, /noerase, /nodata,color='red',symsize=2, position=[0.65,0.38,0.96,0.60],xtitle='Fil-min-length',ytitle='fit angle(deg)',isotropic=0
    a2length = hist_2d(max(angle_table[9:10,*], dimension = 1),bin1=1,min1=0,angle_table[4,*],bin2=1,min2=0)
    cgimage, a2length,position=[0.65,0.38,0.96,0.60],/noerase, ctindex=2,/scale
    print,'Min-angle code:', min(angle_table[9,*]),' Max angle code:',max(angle_table[9,*])
    cghistoplot, angle_table[11,*], binsize=1, /noerase,color='blue', position=[0.075,0.38,0.30,0.60],title='angle-code'

    cgplot, max(angle_table[9:10,*],dimension=1),angle_table[5,*], psym=3, /noerase, /nodata,color='red',symsize=2, position=[0.65,0.72,0.96,0.94],xtitle='Fil-length-max',ytitle='direct angle(deg)',isotropic=0
    dl = hist_2d(max(angle_table[9:10,*],dimension=1),bin1=1,min1=0,angle_table[5,*],bin2=1,min2=0)
    cgimage, dl,position=[0.65,0.72,0.96,0.94],/noerase, ctindex=2,/scale

    cgplot, max(angle_table[9:10,*],dimension=1),angle_table[6,*], psym=3, /noerase, /nodata,color='red',symsize=2, position=[0.075,0.72,0.30,0.94],xtitle='Fil-length-max',ytitle='optim angle(deg)',isotropic=0
    dl = hist_2d(max(angle_table[9:10,*],dimension=1),bin1=1,min1=0,angle_table[6,*],bin2=1,min2=0)
    cgimage, dl,position=[0.075,0.72,0.30,0.94],/noerase, ctindex=2,/scale
    return
  endif

  if keyword_set(statistics) and keyword_set(ntable) then begin
    widget_control,widget_info(event.top, find_by_uname='WID_TABLE_NODES'), get_value=node_table
    print, 'Node analysis'
    nodeintensity = reform(node_table[3,*])
    cghistoplot, nodeintensity, nbins=50, /noerase, color='red',position=[0.37,0.08,0.55,0.34], title='SMLM intensity (avg)'
    numfil = total((node_table[5:12,*] ne 0),1)
    numpores = total((node_table[13:20,*] ne 0),1)
    print, max(numfil), min(numfil), max(numpores), min(numpores)
    ;   cgplot, numfil, numpores,/noerase,position=[0.05,0.08,0.28,0.34], psym=3, color='red',xtitle='N-filaments',ytitle='N-pores'
    ;  return
  endif

  if keyword_set(statistics) and keyword_set(ftable) then begin
    print, 'Filament analysis'
    widget_control,widget_info(event.top, find_by_uname='WID_TABLE_FILAMENTS'), get_value=filament_table
    contourlength = filament_table[8,*]
    vectorlength = sqrt((filament_table[4,*]-filament_table[6,*])^2+ (filament_table[5,*]-filament_table[7,*])^2)
    iangle = filament_table[9,*]
    fangle = filament_table[10,*]
    igrangle = filament_table[11,*]
    fgrangle = filament_table[12,*]
    tortuosity = (contourlength+2)/vectorlength
    cgplot, vectorlength,contourlength, psym=3, /noerase, color='red',symsize=2, position=[0.05,0.08,0.34,0.34],xtitle='End-to-end length (pixels)',ytitle='Contour length (pixels)',/isotropic
    cghistoplot, tortuosity, nbins=50, /noerase, color='red',position=[0.37,0.08,0.55,0.34], title='Tortuosity (contour/end-to-end length)
    cghistoplot, iangle, nbins =60, /noerase, color='red', position = [0.06,0.70,0.32,0.95], title='initial-Angle-fit'
    cghistoplot, fangle, nbins =60, /noerase, color='red', position = [0.40,0.70,0.65,0.95], title='terminal-Angle-fit'
    cghistoplot, igrangle, nbins =60, /noerase, color='red', position = [0.06,0.40,0.32,0.65], title='initial-Angle-direct'
    cghistoplot, fgrangle, nbins =60, /noerase, color='red', position = [0.40,0.40,0.65,0.65], title='terminal-Angle-direct'
    cghistoplot, vectorlength, nbins =60, /noerase, color='red', position = [0.73,0.40,0.98,0.65], title='End-End length'
    cghistoplot, contourlength, nbins =100, /noerase, color='red', position = [0.73,0.70,0.98,0.95], title='contour-length'
    cgplot, contourlength, igrangle, psym = 3, /noerase, color='red', position = [0.72,0.08,0.96,0.34], title='length vs. i-grangle',ytitle='i-grangle'
    return
  endif

  if keyword_set(statistics) and keyword_set(ptable) then begin
    print, 'Pore analysis'
    widget_control,widget_info(event.top, find_by_uname='WID_TABLE_PORES'), get_value=pore_table
    porearea = pore_table[3,*]
    vertices = pore_table[4,*]
    pcode = pore_table[5,*]
    goodpores = where(pcode eq 0,/null)
    cghistoplot, porearea[goodpores], nbins=200, /noerase, color='red',position=[0.07,0.08,0.95,0.34], title='Pore Area'
    cghistoplot, porearea[goodpores], nbins=200, /noerase, color='red',position=[0.25,0.40,0.95,0.95], title='Pore Area (log)',/Log
    return
  endif


end


pro meshwork_mask, event, reset=reset, override=override, get_mask=get_mask, mask=mask, set_mask=set_mask, x=x, y=y,boundcheck=boundcheck, inbound=inbound, index1d = index1d, verbose =verbose
  Common meshwork_dataset, properties, image_set, filamentmask, nodemask, node_table, pore_table, filament_table, angle_table, aster_table
  common node_mask, ImageROI_ptrarr_ptrarr

  if keyword_set(boundcheck) then begin
    nmask = ((boundcheck-1)>0)<(properties.frames-1)
    print, 'nmask:', nmask
    mask = *ImageROI_ptrarr_ptrarr[nmask]
    ;    help, mask
    ;    help, image_stack
    ;mask=*((*ImageROI_ptrarr_ptrarr[nmask])[0])
    if keyword_set(verbose) then cgimage, mask,/noerase,/scale,position=[0.15,0,0.3,0.15]
    incoord = where(mask gt 0, ncount)
    inbound = 0
    if ncount lt 1 then inbound = 0
    inmaskdim = size(mask,/dimensions)
    if keyword_set(index1d) then begin
      if mask[index1d] gt 0 then inbound =1
      print, 'index1d'
      return
    endif
    if ~keyword_set(index1d) then begin
      if mask[round(x),round(y)] gt 0 then inbound = 1
      ;print, 'x:', x,' y:', y,' mask:',mask[round(x),round(y)]
      return
    ENDIF
    RETURN
  endif

  if keyword_set(set_mask) then begin
    print,'Set_mask'
    nmask = ((set_mask-1)>0)<(properties.frames-1)
    inmaskdim = size(mask,/dimensions)
    if (inmaskdim[0] ne properties.xpixels) or (inmaskdim[1] ne properties.ypixels) then begin
      print, 'Incorrect dimension'
      return
    endif
    ImageROI_ptrarr_ptrarr[nmask] = [ptr_new(mask)]
    ;    help, mask
    ;    help, ImageROI_ptrarr_ptrarr
    return
  endif

  if keyword_set(get_mask) then begin
    nmask = ((get_mask-1)>0)<(properties.frames-1)
    mask = *ImageROI_ptrarr_ptrarr[nmask]
    ;mask = *(frameptrarr[0])
    ;mask=*((*ImageROI_ptrarr_ptrarr[nmask])[0])
    ; help, mask
    return
  endif

  if keyword_set(reset) then begin
    if ~keyword_set(override) then begin
      z= dialog_message('Reset all ROI?',/cancel)
      if Z eq 'Cancel' then return
    endif
    blank = bytarr(properties.xpixels,properties.ypixels)
    ImageROI_ptrarr_ptrarr=ptrarr(properties.frames)
    for i = 0, properties.frames-1 do ImageROI_ptrarr_ptrarr[i]=[ptr_new(blank)]
    print,'Reset all mask'
    return
  endif

end

pro meshwork_io_status,event,text=text
  Common meshwork_dataset, properties, image_set, filamentmask, nodemask, node_table, pore_table, filament_table, angle_table, aster_table

  widget_control, widget_info(event.top,find_by_uname='WID_TEXT_IO'), set_value=text

end

pro meshwork_io, event, loadsav=loadsav, savesav=savesav, batch = batch, crop=crop
  Common meshwork_dataset, properties, image_set, filamentmask, nodemask, node_table, pore_table, filament_table, angle_table, aster_table

  if keyword_set(savesav) then begin
    if ~keyword_set(batch) then begin
      filename = Dialog_Pickfile(/write,get_path=fpath,filter=['*meshworks.sav'],title='Export data into *meshworks.sav file')
      if filename eq '' then return
      cd,fpath
      filename=meshwork_AddExtension(filename,'_meshworks.sav')
    endif else filename=batch

    if ~keyword_set(crop) then begin
      save, properties,image_set,filamentmask, nodemask, node_table, pore_table, filament_table, angle_table, aster_table,filename=filename
      if ~keyword_set(batch) then  dummy = dialog_message('Finish saving file!: '+filename) else print,'Finish saving file!: ', filename
      meshwork_io_status,event,text='Finish saving file!: '+filename
      return
    endif else begin
      oldimageset = image_set*1.
      z=meshwork_getzoomfactor()
      oldproperties = {datafile: properties.datafile,timestamp: properties.timestamp,xpixels:properties.xpixels,ypixels:properties.ypixels,frames:1.}

      properties.xpixels=z.xrange
      properties.ypixels=z.yrange
      image_set=image_set[z.xmin:z.xmax,z.ymin:z.ymax,*]

      save, properties,image_set,filamentmask, nodemask, node_table, pore_table, filament_table, angle_table,aster_table,filename=filename
      image_set=oldimageset
      properties= oldproperties
      meshwork_io_status,event,text='Finish exporting cropped area to file!: '+filename
      return
    endelse
    return
  endif

  if keyword_set(loadsav) then begin

    if ~keyword_set(batch) then begin
      filename = Dialog_Pickfile(/read,get_path=fpath,filter=['*meshworks.sav'],title='Select *meshworks.sav file to open')
      if filename eq '' then begin
        print,'filename not recognized', filename
        return
      endif
      cd,fpath
    endif else filename=batch

    print,'opening file: ', filename
    meshwork_io_status,event,text='opening file: '+filename
    if strpos(filename,'_meshworks.sav') ne -1 then restore,filename=filename

    properties.datafile=filename

    if size(node_table,/type) ne 0 then begin
      entry = n_elements(node_table[0,*])
      widget_control,widget_info(event.top, find_by_uname='WID_TABLE_NODES'), table_ysize=entry
      widget_control,widget_info(event.top, find_by_uname='WID_TABLE_NODES'), row_labels=string(indgen(entry)+1)
      widget_control,widget_info(event.top, find_by_uname='WID_TABLE_NODES'), set_value=node_table
    endif

    if size(filament_table,/type) ne 0 then begin
      entry = n_elements(filament_table[0,*])
      widget_control,widget_info(event.top, find_by_uname='WID_TABLE_FILAMENTS'), table_ysize=entry
      widget_control,widget_info(event.top, find_by_uname='WID_TABLE_FILAMENTS'), row_labels=string(indgen(entry)+1)
      widget_control,widget_info(event.top, find_by_uname='WID_TABLE_FILAMENTS'), set_value=filament_table
    endif

    if size(pore_table,/type) ne 0 then begin
      entry = n_elements(pore_table[0,*])
      widget_control,widget_info(event.top, find_by_uname='WID_TABLE_PORES'), table_ysize=entry
      widget_control,widget_info(event.top, find_by_uname='WID_TABLE_PORES'), row_labels=string(indgen(entry)+1)
      widget_control,widget_info(event.top, find_by_uname='WID_TABLE_PORES'), set_value=pore_table
    endif

    if size(angle_table,/type) ne 0 then begin
      entry = n_elements(angle_table[0,*])
      widget_control,widget_info(event.top, find_by_uname='WID_TABLE_ANGLES'), table_ysize=entry
      widget_control,widget_info(event.top, find_by_uname='WID_TABLE_ANGLES'), row_labels=string(indgen(entry)+1)
      widget_control,widget_info(event.top, find_by_uname='WID_TABLE_ANGLES'), set_value=angle_table
    endif

    if size(aster_table,/type) ne 0 then begin
      entry = n_elements(aster_table[0,*])
      widget_control,widget_info(event.top, find_by_uname='WID_TABLE_ASTERS'), table_ysize=entry
      widget_control,widget_info(event.top, find_by_uname='WID_TABLE_ASTERS'), row_labels=string(indgen(entry)+1)
      widget_control,widget_info(event.top, find_by_uname='WID_TABLE_ASTERS'), set_value=aster_table
    endif

    return
  endif

end

pro meshwork_imageset_retrieve, x =x, y=y, dim=dim, smlm=smlm, image=im, dimx=xsize, dimy=ysize, oft=oft, asters = asters
  Common meshwork_dataset, properties, image_set, filamentmask, nodemask, node_table, pore_table, filament_table, angle_table, aster_table

  if ~keyword_set(dim) then dim = 100.
  if size(image_set, /type ) eq 0 then return
  if size(properties, /type ) eq 0 then return
  if ~keyword_set(x) or ~keyword_set(y) then return
  theframe = 0
  if keyword_set(oft) then theframe = 1
  if keyword_set(asters) then theframe = 5

  xmin = (x-dim*0.5)>0
  xmax = (x +dim*0.5)<(properties.xpixels-1)
  ymin = (y-dim*0.5)>0
  ymax = (y +dim*0.5)<(properties.ypixels-1)
  im = dblarr(dim+1,dim+1)
  xsize = xmax-xmin+1
  ysize = ymax-ymin+1
  imtemp = reform(image_set[xmin:xmax,ymin:ymax, theframe])
  im[0:xsize-1,0:ysize-1] = reform(image_set[xmin:xmax,ymin:ymax, theframe])

end

;pro meshwork_imageset_transform, event, updateoft=updateoft, noflip=noflip, noautoshift=noautoshift

;Common meshwork_dataset, properties, image_set, filamentmask, nodemask, node_table, pore_table, filament_table, angle_table, aster_table
;
;if keyword_set(updateoft) then begin
;  if size(properties,/type) ne 8 then return
;
;  if ~keyword_set(noflip) then begin updateoft = reverse(updateoft, 2)
;    cgimage, updateoft,/noerase,/scale,position=[0,0,1.,1.]
;  endif
;
;  oldxsize = properties.xpixels
;  oldysize = properties.ypixels
;  oftxsize = (size(updateoft, /dimensions))[0]
;  oftysize = (size(updateoft, /dimensions))[1]
;
;  newxsize = max([oldxsize,oftxsize])
;  newysize = max([oldysize,oftysize])
;
;  xshift = abs(newxsize-oldxsize)/2.
;  yshift = abs(newysize-oldysize)/2.
;
;  newimageset = dblarr(newxsize,newysize,8)
;  newimageset[0:properties.xpixels-1,0:properties.ypixels-1,0] = image_set[*,*,0]
;  newimageset[0:properties.xpixels-1,0:properties.ypixels-1,1] = image_set[*,*,1]
;  newimageset[0:properties.xpixels-1,0:properties.ypixels-1,2] = image_set[*,*,2]
;  newimageset[0:properties.xpixels-1,0:properties.ypixels-1,3] = image_set[*,*,3]
;  newimageset[0:oftxsize-1,0:oftysize-1,1] = updateoft
;
;  properties.xpixels = newxsize
;  properties.ypixels = newysize
;  frontendzoom = [0,0, properties.xpixels,properties.ypixels]
;
;  if ~keyword_set(noautoshift) then begin
;    newimageset[*,*,0] = shift(newimageset[*,*,0],+1*xshift,+1*yshift)
;    newimageset[*,*,3] = shift(newimageset[*,*,3],+1*xshift,+1*yshift)
;  endif
;
;   image_set=newimageset
;  return
;
;endif

;end

pro meshwork_loadoft, event, image=image, filename = ff, watershed=watershed,xoffset=xoffset,yoffset=yoffset

  Common meshwork_dataset, properties, image_set, filamentmask, nodemask, node_table, pore_table, filament_table, angle_table, aster_table

  if ~keyword_set(xoffset) then xoffset = 10
  if ~keyword_set(yoffset) then yoffset = 10

  if keyword_set(watershed) then begin
    if size(image_set, /type) eq 0 or size(properties, /type) eq 0 then begin
      print, 'No image in memory...'
      z= dialog_message('Please load the SMLM image first...')
      return
    endif

    if ~keyword_set(ff) then begin
      dataFile = Dialog_Pickfile(/read,get_path=fpath,filter=['*.csv','*.csv'],title='Select *.csv file to open')
      if dataFile eq '' then return
      cd, fpath
    endif else dataFile = ff

    if query_csv(dataFile, info) then begin
      help, info
      wscoordinates = dblarr(info.nfields,info.lines)
      oftdata = read_csv(dataFile)
      for i = 0, info.nfields-1 do wscoordinates[i,*]=oftdata.(i)
      help,wscoordinates
      ws = reform(image_set[*,*,0]*0)+1.
      ws[wscoordinates[0,*]+xoffset,wscoordinates[1,*]+yoffset] = 0
      cgimage,ws,/noerase,/scale,position=[0,0,0.2,0.2]
      ws = reverse(ws,2)
      ws = label_region(ws,/all_neighbors)
      cgimage,ws,/noerase,/scale,position=[0.2,0.0,0.4,0.2]
      image_set[*,*,2] = ws
      print,'New watershed image loaded...'
      meshwork_io_status,event,text='New watershed image loaded...' +dataFile
      ;    relabel = meshwork_watershed_relabel(watershedimage=ws)
      ;    cgimage,relabel,/noerase,/scale,position=[0.4,0.0,0.6,0.2]
      return
    endif

    return
  endif


  if ~keyword_set(ff) then begin
    dataFile = Dialog_Pickfile(/read,get_path=fpath,filter=['*.csv','*.csv'],title='Select *.csv file to open')
    if dataFile eq '' then return
    cd, fpath
  endif else dataFile = ff

  if query_csv(dataFile, info) then begin
    help, info
    image = dblarr(info.nfields,info.lines)
    oftdata = read_csv(dataFile)
    for i = 0, info.nfields-1 do image[i,*]=oftdata.(i)
    help,image
    meshwork_io_status,event,text='New OFT image loaded...' +dataFile
    return
  endif

  if query_tiff(dataFile, info) then begin
    help, info
    if info.num_images ne 1 then begin
      print,' Incorrect image numbers'
      z =dialog_message('Only single image can be input')
      return
    endif

    if info.channels eq 3 then begin
      print,' RGB image detected'
      z =dialog_message('RGB image, gray-scale is preferred, Proceed anyway?',/cancel)
      if z eq 'Cancel' then return
      image = read_tiff(dataFile)
      image = total(rawimage, 1)
    endif else image = read_tiff(dataFile)
    return
  endif

end

pro meshwork_loadtiff, event, filename = ff, loadflag= loadflag, noreverse = noreverse, cell=cell, dimension=dimension, image_data=image_data, single=single
  common display_info, def_w, wxsz,wysz, mousedown, infoinitial, frontendwindow, frontendzoom
  Common meshwork_dataset, properties, image_set, filamentmask, nodemask, node_table, pore_table, filament_table, angle_table, aster_table

  if ~keyword_set(ff) then begin
    dataFile = Dialog_Pickfile(/read,get_path=fpath,filter=['*.tif','*.tiff'],title='Select *.tif file to open')
    if dataFile eq '' then return
    cd, fpath
  endif else dataFile = ff

  d =  query_tiff(dataFile, info)
  help, info

  if d ne 1 then begin
    z=dialog_messagE('Invalid file')
    return
  endif

  if info.num_images ne 1 then begin
    print,' Incorrect image numbers'
    z =dialog_message('Only single image can be input')
    return
  endif

  if info.channels eq 3 then begin
    print,' RGB image detected'
    z =dialog_message('RGB image, gray-scale is preferred, Proceed anyway?',/cancel)
    if z eq 'Cancel' then return
    rawimage = read_tiff(dataFile)
    rawimage = total(rawimage, 1)
  endif else rawimage = read_tiff(dataFile)

  help, rawimage
  if size(rawimage,/type) eq 0 then begin
    if keyword_set(loadflag) then loadflag = -1
    return
  endif

  timestamp = (file_info(dataFile)).atime

  if ~keyword_set(noreverse) then rawImage = reverse(rawImage,2,/overwrite)

  dimension = size(rawimage,/dimensions)
  print, dimension

  properties.datafile = datafile
  properties.timestamp = timestamp

  if dimension[0] ne dimension[1] then begin
    print,'Non-squared image', dimension
    z=dialog_message('Image is not a square')
    maxdim = max(dimension)
    newimage = dblarr(maxdim,maxdim)
    marginx = fix((maxdim-dimension[0])*0.5)
    marginy = fix((maxdim-dimension[1])*0.5)
    newimage[marginx:marginx+dimension[0]-1,marginy:marginy+dimension[1]-1]=rawimage
    rawimage=newimage
    dimension = [maxdim,maxdim]
  endif

  properties.xpixels=dimension[0]
  properties.ypixels=dimension[1]

  properties.frames = 1
  image_set = dblarr([size(rawimage,/dimensions),8])
  image_set[*,*,0] = rawimage

  meshwork_mask, event, /reset
  image_transform_parameters = [0,0,0.]

  print,'Load from:', datafile
  print,'timestamp:', timestamp
  print, 'frames:', properties.frames
  frontendzoom = [0,0, properties.xpixels,properties.ypixels]
  meshwork_io_status,event,text='New tiff image loaded...' +dataFile
end

pro meshwork_scalebar, event, size=size
  common display_info, def_w, wxsz,wysz, mousedown, infoinitial, frontendwindow, frontendzoom
  Common meshwork_dataset, properties, image_set, filamentmask, nodemask, node_table, pore_table, filament_table, angle_table, aster_table

  if ~keyword_set(size) then size =2.0
  z = meshwork_getzoomfactor()

  widget_control,Widget_Info(event.top, FIND_BY_UNAME='WID_SLIDER_SMLMPIXELSIZE'),get_value=smlmpixelsize
  smlmpixelsizeum = smlmpixelsize/1000.

  print, 'mgw:', z.mgw
  print, 'xdim:', z.xdim
  screenpixelsize = smlmpixelsize/z.mgw
  print, screenpixelsize ,' nm'

  lengthscalebar = 2.0/screenpixelsize



end

pro meshwork_goto,event, x=x , y=y, winsize=winsize
  common display_info, def_w, wxsz,wysz, mousedown, infoinitial, frontendwindow, frontendzoom
  Common meshwork_dataset, properties, image_set, filamentmask, nodemask, node_table, pore_table, filament_table, angle_table, aster_table

  if ~keyword_set(winsize) then winsize = 50
  if ~keyword_set(x) then x = 100
  if ~keyword_set(y) then y = 100
  if size(properties, /type) eq 0 then return
  if size(image_set, /type) eq 0 then return

  z = meshwork_getzoomfactor()
  help, z
  xmin = (x-winsize*0.5)>0
  xmax = (x+winsize*0.5)<(properties.xpixels-1)
  ymin = (y-winsize*0.5)>0
  ymax = (y+winsize*0.5)<(properties.ypixels-1)

  xdim = xmax-xmin+1 & ydim = ymax-ymin+1

  mgw = ((wxsz*1./xdim)<(wysz*1./ydim))

  frontendzoom = [xmin,ymin,xmax,ymax]
  z = meshwork_getzoomfactor()
  ; help, z

end

pro meshwork_azimuth,event, test=test, image=image, sector=sector, verbose=verbose, selection=selection , blank=blank,noerase=noerase, smlm=smlm, oft=oft
  Common meshwork_dataset, properties, image_set, filamentmask, nodemask, node_table, pore_table, filament_table, angle_table, aster_table

  if ~keyword_set(sector) then sector = 180.
  if ~keyword_set(noerase) then cgerase,'white'
  if ~keyword_set(margin) then margin = 10
  if keyword_set(test) then begin
    testimage1 = dblarr(101,101)
    testimage1 = meshwork_getCircularMask(testimage1,10)*1
    testimage1[48:52,*]=1
    testimage1[*,48:52]=1
    testimage1 = testimage1 or rot(testimage1,45,1.0)
    testimage1 = testimage1 or rot(testimage1,22.5,1.0)
    cgimage, testimage1,/noerase, /scale, position= [0,0,0.25,0.25]

    mask50 = meshwork_getCircularMask(testimage1,50)
    cgimage, mask50,/noerase, /scale, position= [0.25,0,0.50,0.25]

    centerx = 50
    centery = 50
    correl = dblarr(sector)
    secinterval = findgen(sector)*360./60.
    for i = 0, sector-1 do begin
      rotim = rot(testimage1*mask50,i*360./sector,1.0)
      cgimage, rotim,/noerase,/scale,position=[0.5,0.0,0.75,0.25]
      correl[i] = total(testimage1*rotim)
    endfor
    correl = correl/total(testimage1^2)
    ;cgplot, secinterval, correl, /noerase, position=[0.08,0.3,0.95,0.55],axiscolor='red',title='Azimuthal correlation funciton'
    xfft = findgen(sector)/360
    correl[0:margin] = 0
    correl[-1*margin:-1]=0
    powerspectrum = (abs(fft(correl-min(correl))*2))^2/2
    cgplot, secinterval, correl, /noerase, position=[0.08,0.3,0.95,0.55],axiscolor='red',title='Azimuthal correlation funciton'
    cgplot, xfft[0:0.5*sector-1],alog10(powerspectrum[0:0.5*sector-1]), /noerase, position=[0.08,0.62,0.45,0.95],axiscolor='red',title='PowerSpectrum of Azimuthal correlation funciton',ytitle='Log10 Power',color='blue'
    cgoplot, xfft[0:0.5*sector-1],alog10(powerspectrum[0:0.5*sector-1]), psym=2
    cgplot, xfft[0:0.5*sector-1],powerspectrum[0:0.5*sector-1], /noerase, position=[0.58,0.62,0.96,0.95],axiscolor='red',title='PowerSpectrum of Azimuthal correlation funciton',ytitle='Power',color='blue',thick=2
    cgoplot, xfft[0:0.5*sector-1],powerspectrum[0:0.5*sector-1], psym=3
    return
  endif

  if keyword_set(image) and keyword_set(selection) then begin
    if size(properties, /type) eq 0 then return
    if size(image_set, /type) eq 0 then return
    if ~keyword_set(dimension) then dimension=100.
    widget_control,widget_info(event.top, find_by_uname='WID_TABLE_ASTERS'), get_value=aster_table

    help,selection
    if n_elements(selection) lt 1 then return
    meshwork_imageset_retrieve, x =aster_table[1,selection[0]-1], y=aster_table[2,selection[0]-1], smlm=smlm, oft=oft, image=im
    if keyword_set(blank) then meshwork_imageset_retrieve, x =aster_table[1,selection[0]-1], y=aster_table[2,selection[0]-1], /aster, image=astermask $
    else astermask = 1.
    cgimage, im*(~astermask),/scale,/noerase,position=[0,0,0.25,0.25]
    cgplots, 0.125,0.125, psym=2,color='green',symsize=1,/normal,thick=1
    ;help,astermask
    if keyword_set(blank) then    meshwork_azimuth,event, image=im, sector=sector, verbose=verbose, blank=astermask else meshwork_azimuth,event, image=im, sector=sector, verbose=verbose
    return
  endif

  if keyword_set(image) then begin
    print,'Azimuthal analysis..'
    imdim = size(image,/dimensions)
    dimx = imdim[0]
    dimy = imdim[1]
    centerx = dimx*0.5
    centery = dimy*0.5
    correl = dblarr(sector)
    secinterval = findgen(sector)*360./sector
    if ~keyword_set(blank) then blank = image*0+1.
    mask = meshwork_getCircularMask(blank,centerx,/inner)
    imagex=image-min(image)
    if keyword_set(blank) then bmask = ~blank else bmask = 1.
    if keyword_set(verbose) then cgimage, imagex,/noerase,/scale,position=[0.0,0,0.25,0.25]
    if keyword_set(verbose) then cgplots, 0.125,0.125,/normal,psym=2,color='green'
    for i = 0, sector-1 do begin
      rotim = rot(imagex*mask*bmask,secinterval[i],1.0)
      correl[i] = total(imagex*rotim)
      if keyword_set(verbose) then cgimage, rotim,/noerase,/scale,position=[0.25,0,0.50,0.25]
    endfor
    correl = correl/total(imagex^2)
    ;correl=correl-min(correl)
    cgplot, secinterval, correl, /noerase, position=[0.08,0.3,0.45,0.55],axiscolor='red',title='Azimuthal auto-correlation funciton',yrange=[0,1],xstyle=1
    cgplot, secinterval, correl-min(correl), /noerase, position=[0.58,0.3,0.96,0.55],axiscolor='red',title='Osc. components. of AAF',yrange=[0,1],xstyle=1

    ;cgoplot, secinterval, correl, psym=2
    xfft = findgen(sector)/360
    powerspectrum = (abs(fft(correl)*2))^2/2

    cgplot, xfft[5:0.5*sector-1],alog10(powerspectrum[5:0.5*sector-1]), /noerase, position=[0.08,0.62,0.45,0.95],axiscolor='red',title='PowerSpectrum of Azimuthal correlation funciton',ytitle='Log10 Power',color='blue', $
      xstyle=1
    cgoplot, xfft[5:0.5*sector-1],alog10(powerspectrum[5:0.5*sector-1]), psym=2
    cgplot, xfft[5:0.5*sector-1],sqrt(powerspectrum[5:0.5*sector-1]), /noerase, position=[0.58,0.62,0.96,0.95],axiscolor='red',title='PowerSpectrum of Azimuthal correlation funciton',ytitle='Power',color='blue',thick=2,$
      xstyle=1
    cgoplot, xfft[5:0.5*sector-1],sqrt(powerspectrum[5:0.5*sector-1]), psym=3

    flcorrel = min(correl)
    print,'Osc.ratio:', mean((correl-flcorrel)/flcorrel)

    correl[0:margin] = 0
    correl[-1*margin:-1]=0
    print,'De-margined Osc.ratio:', mean((correl[margin+1:-1*margin-1]-flcorrel)/flcorrel)
    print,'total ratio (de-margined):', total((correl[margin+1:-1*margin-1]-flcorrel)/flcorrel)
    return
  endif

end


function meshwork_getzoomfactor, xgoto=xgoto, ygoto=ygoto
  common display_info, def_w, wxsz,wysz, mousedown, infoinitial, frontendwindow, frontendzoom
  Common meshwork_dataset, properties, image_set, filamentmask, nodemask, node_table, pore_table, filament_table, angle_table, aster_table

  ;    if keyword_set(xgoto) and keyword_set(ygoto) then begin
  ;      xmin = frontendzoom[0]>0
  ;      ymin = frontendzoom[1]>0
  ;      xmax = frontendzoom[2]<(properties.xpixels-1)
  ;      ymax = frontendzoom[3]<(properties.ypixels-1)
  ;      xdim = xmax-xmin+1 & ydim = ymax-ymin+1
  ;
  ;      xgoto = (xgoto>0)<(properties.xpixels-1)
  ;      ygoto = (ygoto>0)<(properties.ypixels-1)
  ;
  ;      xmin = (xgoto-0.5*xdim)>0
  ;      xmax = (xgoto+0.5*xdim) >0
  ;      ymin = (ygoto-0.5*ydim) <(properties.xpixels-1)
  ;      ymax = (ygoto+0.5*ydim)<(properties.ypixels-1)
  ;      xdim = xmax-xmin+1 & ydim = ymax-ymin+1
  ;      mgw = ((wxsz*1./xdim)<(wysz*1./ydim))
  ;
  ;      frontendzoom = [xmin,ymin,xmax,ymax]
  ;
  ;      return, {mgw:mgw,xmin:xmin,ymin:ymin,xrange:xdim,yrange:ydim,xmax:xmax,ymax:ymax,wxsz:wxsz,wysz:wysz,xdim:xdim,ydim:ydim}
  ;    endif

  xmin = frontendzoom[0]>0
  ymin = frontendzoom[1]>0
  xmax = frontendzoom[2]<(properties.xpixels-1)
  ymax = frontendzoom[3]<(properties.ypixels-1)
  xdim = xmax-xmin+1 & ydim = ymax-ymin+1
  mgw = ((wxsz*1./xdim)<(wysz*1./ydim))
  return, {mgw:mgw,xmin:xmin,ymin:ymin,xrange:xdim,yrange:ydim,xmax:xmax,ymax:ymax,wxsz:wxsz,wysz:wysz,xdim:xdim,ydim:ydim, xc:0.5*(xmin+xmax),yc:0.5*(ymin+ymax)}

end


function meshwork_StringTokenizer, incometext,delimiter=delimiter
  if ~keyword_set(delimiter) then delimiter = " ,"
  thenumber = strsplit(incometext,delimiter,/extract,count=count)
  return, uint(thenumber)
end

pro meshwork_savescreentiff, event
  filename = Dialog_Pickfile(/write,get_path=fpath)
  if strlen(fpath) ne 0 then cd,fpath
  if filename eq '' then return
  presentimage=reverse(tvrd(true=1),3)
  filename=AddExtension(filename,'.tiff')
  write_tiff,filename,presentimage,orientation=1
end

function meshwork_getCircularMask, im, maskrad,  inner=inner

  imdim = size(im,/dimensions)
  xdim = imdim[0]
  ydim = imdim[1]
  xcoord = findgen(xdim)
  ycoord = findgen(ydim)
  centerx = fix(xdim*0.5)
  centery = fix(ydim*0.5)
  distance = dblarr(imdim)
  for i = 0, xdim-1 do for j = 0, ydim-1 do begin
    distance[i,j] = sqrt((centerx-xcoord[i])^2+(centery-ycoord[j])^2)
  endfor

  if keyword_set(inner) then begin
    wheremask  = where(im gt 0, count)
    if count gt 1 then begin
      indexmask = array_indices(im, wheremask)
      disttocentermax = max(sqrt((indexmask[0]-centerx)^2+(indexmask[1]-centery)^2))
      wheremask = where((distance ge maskrad) or (distance le disttocentermax))
      result = im*0+1.
      result[wheremask] = 0
      return, result
    endif
  endif

  wheremask = where(distance ge maskrad)
  result = im*0+1.
  result[wheremask] = 0
  return, result
end

function meshwork_uppertriangle, n, nan=nan
  i = REBIN(LINDGEN(n), n, n)
  j = REBIN(TRANSPOSE(LINDGEN(n)), n, n)
  if keyword_set(nan) then begin
    upp = (i GE j)
    return, 1./upp
  endif else return, (i GE j)
end

function meshwork_areaPolygon,x,y
  ;calculate area using determinants
  N=n_elements(x)
  xx = [x, x[0], x[1]]
  yy = [y, y[0], y[1]]
  area = 0.
  for i = 0, N do area=area+ xx[i]*( yy[i+1] - yy[i-1])
  return,area/2.
end

function meshwork_AddExtension, filename, extension    ; checks if the filename has the extension (caracters after dot in "extension" variable. If it does not, adds the extension.
  sep = !VERSION.OS_family eq 'unix' ? '/' : '\'
  dot_pos=strpos(extension,'.',/REVERSE_OFFSET,/REVERSE_SEARCH)
  short_ext=strmid(extension,dot_pos)
  short_ext_pos=strpos(filename,short_ext,/REVERSE_OFFSET,/REVERSE_SEARCH)
  ext_pos=strpos(filename,extension,/REVERSE_OFFSET,/REVERSE_SEARCH)
  add_ext=(ext_pos lt 0)  ? extension : ''
  file_sep_pos=strpos(filename,sep,/REVERSE_OFFSET,/REVERSE_SEARCH)
  file_dot_pos=strpos(filename,'.',/REVERSE_OFFSET,/REVERSE_SEARCH)
  filename_without_ext = (file_dot_pos gt file_sep_pos) ? strmid(filename,0,file_dot_pos) :  filename
  filename_with_ext = (ext_pos gt 0) ? filename : (filename_without_ext + add_ext)
  return,filename_with_ext
end


function meshwork_StripExtension, filename   ; checks if the filename has the extension (caracters after dot in "extension" variable. If it does not, adds the extension.
  sep = !VERSION.OS_family eq 'unix' ? '/' : '\'
  file_sep_pos=strpos(filename,sep,/REVERSE_OFFSET,/REVERSE_SEARCH)
  file_dot_pos=strpos(filename,'.',/REVERSE_OFFSET,/REVERSE_SEARCH)
  filename_without_ext = (file_dot_pos gt file_sep_pos) ? strmid(filename,0,file_dot_pos) :  filename
  return,filename_without_ext
end

function meshwork_filaments_sort, xi=xi,yi=yi,xf=xf,yf=yf, xpoints=xpoints,ypoints=ypoints, tangent=tangent, boxcar=boxcar, cutoff=cutoff

  if ~keyword_set(tangent) then tangent = 5
  if ~keyword_set(cutoff) then cutoff = 10
  npoints =n_elements(xpoints)

  currentX = xi
  currentY = yi
  status = intarr(npoints)+1;              %track if a points been seen already
  numInitPts      = npoints;

  sortx = xpoints*0-1.
  sorty = ypoints*0-1.
  sortorder = xpoints*0-1.

  i = 0
  while 1 do begin ;
    leftoverpoints = where(status gt 0, count)
    if count lt 1 then break

    sqdisttoall = (currentX - xpoints[leftoverpoints])^2 + (currentY-ypoints[leftoverpoints])^2
    nearest = min(sqdisttoall, wheremin)
    sortorder[i] = leftoverpoints[wheremin]
    status[leftoverpoints[wheremin]] = -1
    currentX = xpoints[leftoverpoints[wheremin]]
    currentY = ypoints[leftoverpoints[wheremin]]
    i++
  endwhile

  newx = xpoints[sortorder]
  newy = ypoints[sortorder]

  if npoints gt cutoff then begin
    xsm = ts_smooth([xi, newx, xf],boxcar,/double)
    ysm = ts_smooth([yi, newy, yf],boxcar,/double)
  endif else begin
    xsm = [xi, newx, xf]
    ysm = [yi, newy, yf]
    tangent = 2
  endelse

  itipvector = [mean(xsm[0:tangent])-xi,mean(ysm[0:tangent])-yi]
  ftipvector = [mean(xsm[-(1+tangent):-1])-xf,mean(ysm[-(1+tangent):-1])-yf]

  ifit = linfit(xsm[0:tangent],ysm[0:tangent])
  ffit = linfit(xsm[-(1+tangent):-1],ysm[-(1+tangent):-1])

  itip = meshwork_getangle(vector=itipvector, slope = ifit[1], xi=xi,xf=xf,yi=yi,yf=yf)
  ftip = meshwork_getangle(vector=ftipvector, slope = ffit[1], xi=xi,xf=xf,yi=yi,yf=yf)


  igrangle = itip.directangle
  fgrangle = (180+igrangle) mod 360


  return, {x: newx,y: newy, itangent:ifit, ftangent:ffit, iquadrant:itip.quadrant, fquadrant: ftip.quadrant, $
    iangle:itip.angle,fangle:ftip.angle, xsmooth:xsm, ysmooth:ysm, igrangle:igrangle, fgrangle:fgrangle, idquadrant:itip.directquadrant, fdquadrant:ftip.directquadrant}

end

function meshwork_mixangle,a1=a1,a2=a2, weight=weight
  if ~keyword_set(weight) then weight = 0.5

  if n_elements(a1) gt 1 then begin
    results = a1*0.
    for i = 0, n_elements(a1)-1 do begin
      if abs(a1[i]-a2[i]) le 180 then results[i] = weight*a1[i]+(1-weight)*a2[i] else results[i] = (180+weight*a1[i]+(1-weight)*a2[i]) mod 360
    endfor
    return, results
  endif

  if abs(a1-a2) le 180 then return, weight*a1+(1-weight)*a2

  return, ((weight*a1+(1-weight)*a2)+180.) mod 360

end

function meshwork_diff_angle, a1=a1,a2=a2, cw=cw, ccw=ccw

  if keyword_set(ccw) then if a2 gt a1 then return, (a2-a1) mod 360 else return, (360-(a1-a2)) mod 360

  if a1 gt a2 then return, (a1-a2) mod 360 else return, (360-(a2-a1)) mod 360
end


function meshwork_makemontage, image_stack=image, auto=auto, x=x, y=y,transpose=transpose, column = column,wxsize=wxsize,wysize=wysize, panels=panels

  if size(image,/n_dimensions) ne 3 then return, -1
  stack = image
  if keyword_set(transpose) then stack = transpose(stack,[1,0,2])
  dimx = (size(stack,/dimensions))[0]
  dimy = (size(stack,/dimensions))[1]
  dimf = (size(stack,/dimensions))[2]

  if keyword_set(auto) then begin
    dim = fix(sqrt(dimf))+1
    x = dim
    y = dim
  end

  if keyword_set(column) then begin
    y = ((dimf mod column)eq 0 )?  fix(dimf/column):(fix(dimf/column)+1)
    x = column
  end

  if keyword_set(x) and ~keyword_set(y) then y=ceil(dimf/x)
  if keyword_set(y) and ~keyword_set(x) then x=ceil(dimf/y)

  montimage = dblarr(dimx*x,dimy*y)
  wxsize = dimx*x
  wysize=dimy*y
  xoffset = (findgen(dimf) mod x)
  yoffset = (y-1-fix(findgen(dimf)/ x))
  panels=[x,y]
  for i = 0, dimf -1 do montimage[xoffset[i]*dimx:xoffset[i]*dimx+dimx-1,yoffset[i]*dimy:yoffset[i]*dimy+dimy-1]=stack[*,*,i]

  return, montimage
end

function meshwork_getangle, vector=vector, slope=slope, directangle=directangle, xi=xi,yi=yi,xf=xf,yf=yf
  if ~keyword_set(slope) then slope = vector[1]/vector[0]
  angle = 1./0
  if (vector[0] eq 0)  or ~finite(vector[0]) then if vector[1] gt 0 then angle = 90. else angle = 270.
  if (vector[1] eq 0)  or ~finite(vector[1]) then if vector[0] gt 0 then angle = 0. else angle = 180.

  quadrant = 1/0.
  if ~finite(angle) then begin
    angle = atan(slope)/!dtor
    ; print,'atan angle:',angle
    if vector[0] gt 0 and vector [1] gt 0 then quadrant = 1
    if vector[0] gt 0 and vector [1] lt 0 then quadrant = 4
    if vector[0] lt 0 and vector [1] lt 0 then quadrant = 3
    if vector[0] lt 0 and vector [1] gt 0 then quadrant = 2
    case quadrant of
      1: if angle lt 0 then if abs(angle) gt 45 then quadrant =2 else quadrant = 4
      2: if angle gt 0 then if abs(angle) gt 45 then quadrant = 1 else quadrant = 3
      3: if angle lt 0 then if abs(angle) gt 45 then quadrant = 4 else quadrant = 2
      4: if angle gt 0 then if abs(angle) gt 45 then quadrant = 3 else quadrant = 1
    endcase

    case quadrant of
      1:
      2: angle = angle+180.
      3: angle = 180+angle
      4: angle = 360.+angle
    endcase
  endif

  angle = angle gt 0? angle: 360.+angle
  angle = angle mod 360

  ;===directangle
  dquadrant = 1/0.
  directangle = 1./0

  if (xf-xi eq 0)   then if yf-yi gt 0 then directangle = 90. else directangle = 270.
  if (yf-yi eq 0)   then if xf-xi gt 0 then directangle = 0. else directangle = 180.


  if ~finite(directangle)  then begin
    directangle = (atan((yf-yi)/(xf-xi))/!DTOR)
    if (xf-xi) gt 0 and (yf-yi) gt 0 then dquadrant = 1
    if (xf-xi) gt 0 and (yf-yi) lt 0 then dquadrant = 4
    if (xf-xi) lt 0 and (yf-yi) lt 0 then dquadrant = 3
    if (xf-xi) lt 0 and (yf-yi) gt 0 then dquadrant = 2
    case dquadrant of
      1: if directangle lt 0 then if abs(directangle) gt 45 then dquadrant =2 else dquadrant = 4
      2: if directangle gt 0 then if abs(directangle) gt 45 then dquadrant = 1 else dquadrant = 3
      3: if directangle lt 0 then if abs(directangle) gt 45 then dquadrant = 4 else dquadrant = 2
      4: if directangle gt 0 then if abs(directangle) gt 45 then dquadrant = 3 else dquadrant = 1
    endcase

    case dquadrant of
      1:
      2: directangle = directangle+180.
      3: directangle = 180+directangle
      4: directangle = 360.+directangle
    endcase

  endif


  return, {angle:angle, quadrant:quadrant, directquadrant:dquadrant,directangle:directangle}
end

FUNCTION meshwork_Arc, xcenter, ycenter, radius, angle1, angle2
  ;Coyote library

  IF N_Elements(angle1) EQ 0 THEN angle1 = 0.0
  IF N_Elements(angle2) EQ 0 THEN angle2 = 360.0
  points = (2 * !DPI  * !RaDeg / 999.0) * Findgen(1000)
  indices = Where((points GE angle1) AND (points LE angle2), count)
  IF count GT 0 THEN points = points[indices]
  x = xcenter + radius * COS(points * !DtoR )
  y = ycenter + radius * SIN(points * !DtoR)
  RETURN, Transpose([[x],[y]])
END

pro matlab_hmin, image=image, h=h, matlab=matlab, results=results

  if ~keyword_Set(matlab) then matlab='IDLcomIDispatch$ProgID$Matlab_Application_23.2' else matlab = 'IDLcomIDispatch$ProgID$'+matlab


  oMatlab = obj_new(matlab)
  oMatlab.PutWorkspaceData, 'im', 'base', image
  oMatlab.PutWorkspaceData, 'h', 'base', h
  oMatlab.Execute, 'c = imhmin(im,h)'
  oMatlab.GetWorkspaceData, 'c', 'base', results
  obj_destroy, oMatlab
end

pro matlab_watershed, image=image, matlab=matlab, results=results

  if ~keyword_Set(matlab) then matlab='IDLcomIDispatch$ProgID$Matlab_Application_23.2' else matlab = 'IDLcomIDispatch$ProgID$'+matlab

  oMatlab = obj_new('IDLcomIDispatch$ProgID$Matlab_Application_23.2')
  oMatlab.PutWorkspaceData, 'im', 'base', image
  oMatlab.Execute, 'c = watershed(im)'
  oMatlab.GetWorkspaceData, 'c', 'base', results
  obj_destroy, oMatlab
end

pro matlab_OFT, image=image, radius=radius,sector=sector, OFT=OFT, LFT=LFT, Orientation = Orientation, padding=padding, squarecrop = squarecrop, verbose=verbose, $
  matlab=matlab

  if ~keyword_Set(matlab) then matlab='IDLcomIDispatch$ProgID$Matlab_Application_23.2' else matlab = 'IDLcomIDispatch$ProgID$'+matlab

  if ~keyword_seT(image) then return
  if ~keyword_set(sector) then sector = 20.
  if ~keyword_set(radius) then radius = 10.

  if keyword_set(squarecrop) then begin
    imdim = size(image,/dimensions)
    if imdim[0] ne imdim[1] then if imdim[0] gt imdim[1] then image=image[0:imdim[1]-1,*] else image=image[*,0:imdim[0]-1]
  endif

  imdim = size(image,/dimensions)
  if keyword_set(padding) then begin
    newxdim = imdim[0]+2*radius
    newydim = imdim[1]+2*radius
    newim = dblarr(newxdim, newydim)
    newim[radius:radius+imdim[0]-1,radius:radius+imdim[1]-1] = image
    theimage = newim
  endif

  mask = theimage*0.+1.
  oMatlab = obj_new(matlab)
  oMatlab.PutWorkspaceData, 'image', 'base', theimage
  oMatlab.PutWorkspaceData, 'radius', 'base', radius
  oMatlab.PutWorkspaceData, 'sector', 'base', sector
  oMatlab.PutWorkspaceData, 'mask', 'base', mask
  
  ; add current path ot matlab path @qiqilu
  thisFile = ROUTINE_FILEPATH()
  thisPath = FILE_DIRNAME(thisFile)
  cmd = "addpath('"+thisPath+"');"
  oMatlab.Execute, cmd
  
  oMatlab.Execute, '[OFT_Img, LFT_Img, LFT_Orientations] = LFT_OFT_mex(double(image),double(radius),double(sector),double(mask));'
  oMatlab.GetWorkspaceData, 'OFT_Img', 'base', OFT
  oMatlab.GetWorkspaceData, 'LFT_Img', 'base', LFT
  oMatlab.GetWorkspaceData, 'LFT_Orientations', 'base', Orientation

  if keyword_set(padding) then begin
    ;print, 'Remove padding'
    ;help, OFT, radius, imdim
    OFT = OFT[radius:radius+imdim[0]-1,radius:radius+imdim[1]-1]
    ;help, OFT
    LFT = LFT[radius:radius+imdim[0]-1,radius:radius+imdim[1]-1]
    Orientation = Orientation[radius:radius+imdim[0]-1,radius:radius+imdim[1]-1]
  endif

  if keyword_set(verbose) then help, OFT, LFT, Orientation
  obj_destroy, oMatlab
end


FUNCTION meshwork_RepMat, matrix, ncol, nrow
  ; David Fanning 2009
  ; On error, return to caller.
  ON_ERROR, 2

  ; Check parameters.
  IF N_Elements(matrix) EQ 0 THEN Message, 'Must pass an array or matrix to replicate.'
  IF N_Params() EQ 2 THEN nrow = ncol ; Number of columns and rows is the same.
  IF N_Elements(ncol) EQ 0 THEN ncol = 1
  IF N_Elements(nrow) EQ 0 THEN nrow = 1

  s = Size(matrix, /DIMENSIONS)
  IF N_Elements(s) EQ 1 THEN s = [s,1] ; Handle the case of a vector being passed in

  ; Create array.
  array = Make_Array(s[0]*ncol, s[1]*nrow, TYPE=Size(matrix,/TYPE))
  array[0,0] = matrix

  ; Replicate rows first.
  IF nrow GT 1 THEN BEGIN
    FOR nrow=1,nrow-1 DO $
      array[0,nrow*s[1]] = matrix
  ENDIF

  ; Replicate columns next.
  IF ncol GT 1 THEN BEGIN
    rmatrix = array[0:s[0]-1,*]
    FOR ncol=1,ncol-1 DO $
      array[(ncol*s[0]),0] = rmatrix
  ENDIF

  RETURN, array
END

;pro meshwork_chisqtest,event
;
;  file1 = Dialog_Pickfile(/read,get_path=fpath,filter=['*histo.csv','*histo.csv'],title='Select 1st header-less *histo.csv file to open')
;  if file1 eq '' then return
;  cd, fpath
;
;  file2 = Dialog_Pickfile(/read,get_path=fpath,filter=['*histo.csv','*histo.csv'],title='Select 2nd header-less *histo.csv file to open')
;  if file2 eq '' then return
;  cd, fpath
;
;  meshwork_io_status,event, text='File:'+file1+ ' and File:'+file2
;
;
;  if query_csv(file1, info) then begin
;    help, info
;    data1 = dblarr(info.nfields,info.lines)
;    f1 = read_csv(file1)
;    for i = 0, info.nfields-1 do data1[i,*]=f1.(i)
;  endif
;
;  if query_csv(file2, info) then begin
;    help, info
;    data2 = dblarr(info.nfields,info.lines)
;    f2 = read_csv(file2)
;    for i = 0, info.nfields-1 do data2[i,*]=f2.(i)
;  endif
;
;  cgerase,'white'
;  cgplot,data1[0,*],data1[1,*],psym=10,color='red',title='Red:'+file1+' Blue:'+file2,thick=2,yrange=[0,max([reform(data1[1,*]),reform(data2[1,*])])]
;  cgoplot,data2[0,*],data2[1,*],psym=10,color='blue',thick=2
;
;  szdata1 = n_elements(data1[1,*])
;  szdata2 = n_elements(data2[1,*])
;  maxsz = max([szdata1,szdata2])
;  resizedata1 = dblarr(maxsz)
;  resizedata2 = dblarr(maxsz)
;  resizedata1[0:szdata1-1]=data1[1,*]
;  resizedata2[0:szdata2-1]=data2[1,*]
;  resizedata1=resizedata1/total(resizedata1)
;   resizedata2=resizedata2/total(resizedata2)
; ; chisq = xsq_test(resizedata2,resizedata1, excell=resizedata1,obcell=resizedata2)
;  ;print, chisq
;  mychisq = total((resizedata1-resizedata2)^2/resizedata1,/nan)
;  anotherchisq = total((resizedata1-resizedata2)^2/(resizedata1+resizedata2),/nan)
;  print,'The Chisq:',mychisq,' Dof:',maxsz-5
;  print,'Alt. Chisq:',anotherchisq
;  pval = chisqr_pdf(mychisq,maxsz-5)
;  print,'The p-value:',pval
;  cgtext, 0.5,0.35,'Chi-Sq p-value:'+string(pval,format='(E20.7)'),/normal,color='dark green',charsize=2
;  cgtext, 0.5,0.25,'Chi^2 .:'+string(mychisq,format='(F12.7)'),/normal,color='dark green',charsize=2
;  cgtext, 0.5,0.20,'Dof:'+string(maxsz-5, format='(I10)'),/normal,color='dark green',charsize=2
;
;
;
;end

