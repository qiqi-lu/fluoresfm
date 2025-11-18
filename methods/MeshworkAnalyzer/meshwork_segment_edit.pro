function meshwork_segment_edit, image, groupleader=groupleader, grayscale = refimage, stack=stack
  common sfa, faimage, grayimage, binaryimage, background, iscancel, sfawindow, oldwindow, stackmode
  common dataset, properties, stackraw, stackproc, auxilliary, driftparameters, shiftparameters
  common famask, famaskbinary
  common cellmask, cellmaskbinary, distancemap, averageimage, cellmaskgrayscale, cellmaskdynamic

  stackmode = 0
  if keyword_set(stack) then begin
    if size(stackproc, /type) eq 0 then return, -1
    if size(famaskbinary, /type) eq 0 then return, -1
    if size(cellmaskdynamic, /type) eq 0 then return, -1

    stackmode = stack
    faimage = stackproc[*,*,0]
    grayimage = stackproc[*,*,0]
    oldwindow = !d.window

    case stack of
      1: binaryimage = famaskbinary[*,*,0]
      2: binaryimage = cellmaskdynamic[*,*,0]
    endcase
    WID_BASE_sfb, GROUP_LEADER=groupleader, _EXTRA=_VWBExtra_,stack=stack
    return, binaryimage
  endif

  binaryimage = image

  if keyword_set(refimage) then grayimage = refimage

  if keyword_set(groupleader) then begin
    oldwindow = !d.window
    WID_BASE_sfb, GROUP_LEADER=groupleader, _EXTRA=_VWBExtra_
    return, binaryimage
  endif else return, -1

end


function splitregions, binaryimage, grayimage, n=n, keepmax = keepmax, split=split

  if ~keyword_set(n) then n = 2
  if size(binaryimage,/type) eq 0 then return, -1
  if size(grayimage,/type) eq 0 then return, -1

  im = binaryimage*grayimage
  index = where(binaryimage, count)
  data = dblarr(3,count)
  data[0:1,*] = array_indices(binaryimage,index)
  data[2,*] = reform(im[index])

  min0 = min(data[0,*],max=max0) & data[0,*] = (data[0,*]-min0)/(max0-min0)
  min1 = min(data[1,*],max=max1) & data[1,*] = (data[1,*]-min1)/(max1-min1)
  min2 = min(data[2,*],max=max2) & data[2,*] = (data[2,*]-min2)/(max2-min2)

  results = CLUSTER(data, CLUST_WTS(data, N_CLUSTERS = n,variable_wts=[0.0,0.0,1]), N_CLUSTERS = n)

  if keyword_set(n) then if n gt 2 then begin
    im=0*binaryimage
    for i = 0, n-1 do im[where(results eq i)] = i
    return, im
  endif

  ;help, results
  results = reform(results)
  ;print, max(results),min(results)
  if (max(results) - min(results)) eq 0 then return, binaryimage else begin

    if keyword_set(keepmax) then begin
      clust0 = where(results eq 0, count0)
      clust1 = where(results eq 1, count1)
      avg0=mean(grayimage[index[clust0]])
      avg1=mean(grayimage[index[clust1]])
      clusthi = (avg0 gt avg1)? clust0:clust1
      im0 = binaryimage*0
      im0[index[clusthi]]=1
      return, im0
    endif

    if keyword_set(split) then begin
      strel = replicate(1,3,3)
      im0 = binaryimage*0
      im1 = binaryimage*0
      im0[where(results eq 0)]=1
      im1[where(results eq 1)]=1
      imor = im1 or im0
      imor[where(dilate(im0,strel) and dilate(im1,strel))]=0
      return, imor
    endif


  endelse

  return, binaryimage

end

pro stackprocess, event, xx, yy, imodex=imodex, threshold=threshold, edit=edit, subtract=subtract
  common sfa, faimage, grayimage, binaryimage, background, iscancel, sfawindow, oldwindow, stackmode
  common interactive, imode, drawwindow, undoimage
  common famask, famaskbinary
  common cellmask, cellmaskbinary, distancemap, averageimage, cellmaskgrayscale, cellmaskdynamic
  common dataset, properties, stackraw, stackproc, auxilliary, driftparameters, shiftparameters

  if size(properties,/type) eq 0 then return
  if size(cellmaskdynamic,/type) eq 0 then return
  if size(famaskbinary,/type) eq 0 then return
  if size(stackproc,/type) eq 0 then return
  if properties.frames lt 1 then return
  if ~keyword_set(imodex) then imodex = 0
  if (imodex eq 7) or (imodex eq 9) and ~keyword_set(threshold) then return

  widget_control,widget_info(event.top,find_by_uname='WID_SLIDER_FRAME'),get_value=theframe
  if ~keyword_set(edit) then case imodex of
  2: begin
    for i = 0, properties.frames-1 do begin
      case stackmode of
        1: famaskbinary[*,*,i] = famaskbinary[*,*,i] or getfloodfillimage(stackproc[*,*,i], xx, yy, threshold)
        2: cellmaskdynamic[*,*,i] = cellmaskdynamic[*,*,i] or getfloodfillimage(stackproc[*,*,i], xx, yy, threshold)
      endcase
    endfor
  end
  3: begin
    for i = 0, properties.frames-1 do begin
      case stackmode of
        1: famaskbinary[*,*,i] = famaskbinary[*,*,i] and ~getfloodfillimage(stackproc[*,*,i], xx, yy, threshold)
        2: cellmaskdynamic[*,*,i] = cellmaskdynamic[*,*,i] and ~getfloodfillimage(stackproc[*,*,i], xx, yy, threshold)
      endcase
    endfor
  end
  4: begin
    for i = 0, properties.frames-1 do begin
      case stackmode of
        1: famaskbinary[*,*,i] = famaskbinary[*,*,i] or ~getfloodfillimage(stackproc[*,*,i], xx, yy, threshold)
        2: cellmaskdynamic[*,*,i] = cellmaskdynamic[*,*,i] or ~getfloodfillimage(stackproc[*,*,i], xx, yy, threshold)
      endcase
    endfor
  end
  5:   begin
    for i = 0, properties.frames-1 do begin
      case stackmode of
        1: famaskbinary[*,*,i] = famaskbinary[*,*,i] xor getselectedregionimage(famaskbinary[*,*,i], xx, yy)
        2: cellmaskdynamic[*,*,i] = cellmaskdynamic[*,*,i] xor getselectedregionimage(cellmaskdynamic[*,*,i], xx, yy)
      endcase
    endfor
  end
  6: begin
    for i = 0, properties.frames-1 do begin
      case stackmode of
        1: famaskbinary[*,*,i] = famaskbinary[*,*,i] or getselectedregionimage(famaskbinary[*,*,i], xx, yy,/grow)
        2: cellmaskdynamic[*,*,i] = cellmaskdynamic[*,*,i] or getselectedregionimage(cellmaskdynamic[*,*,i], xx, yy,/grow)
      endcase
    endfor
  end
  7: begin
    for i = 0, properties.frames-1 do begin
      case stackmode of
        1: famaskbinary[*,*,i] = combinebinary(famaskbinary[*,*,i],getselectedregionimage(famaskbinary[*,*,i], datax, datay,/shrink),update=3)
        2: cellmaskdynamic[*,*,i] = combinebinary(cellmaskdynamic[*,*,i],getselectedregionimage(cellmaskdynamic[*,*,i], datax, datay,/shrink),update=3)
      endcase
    endfor
  end
  8:begin
    for i = 0, properties.frames-1 do begin
      case stackmode of
        1: famaskbinary[*,*,i] = combinebinary(famaskbinary[*,*,i],getselectedregionimage(famaskbinary[*,*,i], datax, datay,/smoothgrow),update=3)
        2: cellmaskdynamic[*,*,i] = combinebinary(cellmaskdynamic[*,*,i],getselectedregionimage(cellmaskdynamic[*,*,i], datax, datay,/shrink),update=3)
      endcase
    endfor
  end
  9:begin
    for i = 0, properties.frames-1 do begin
      case stackmode of
        1: famaskbinary[*,*,i] = combinebinary(famaskbinary[*,*,i],getselectedregionimage(famaskbinary[*,*,i], datax, datay,/smoothshrink),update=3)
        2: cellmaskdynamic[*,*,i] = combinebinary(cellmaskdynamic[*,*,i],getselectedregionimage(cellmaskdynamic[*,*,i], datax, datay,/shrink),update=3)
      endcase
    endfor
  end
  else:
endcase

if keyword_set(edit) then begin
  for i = 0, properties.frames-1 do begin
    case stackmode of
      1: if ~keyword_set(subtract) then famaskbinary[*,*,i] = famaskbinary[*,*,i] or edit else famaskbinary[*,*,i] = famaskbinary[*,*,i] and ~edit
      2: if ~keyword_set(subtract) then cellmaskdynamic[*,*,i] = cellmaskdynamic[*,*,i] or edit else cellmaskdynamic[*,*,i] = cellmaskdynamic[*,*,i] and ~edit
    endcase
  endfor
endif

case stackmode of
  1: binaryimage =famaskbinary[*,*,theframe]
  2: binaryimage =cellmaskdynamic[*,*,theframe]
end
grayimage = stackproc[*,*,theframe]
;showgrayoverlay, event
end


function getselectedregionimage, binaryimage, xx, yy, shrink = shrink, grow=grow, smoothgrow=smoothgrow, smoothshrink = smoothshrink
  labelim = label_region(binaryimage)
  thislabel = labelim[xx,yy]
  results = binaryimage*0

  strel = replicate(1,3,3)
  if thislabel ne 0 then begin
    whererem = array_indices(labelim,where(labelim eq thislabel, pixrem))
    results[whererem[0,*],whererem[1,*]] = 1

    if keyword_set(shrink) then return, erode(results,strel)
    if keyword_set(grow) then return, dilate(results,strel)
    if keyword_set(smoothgrow) then return, morph_close(results,strel)
    if keyword_set(smoothshrink) then return, morph_open(results,strel)


    return, results
  endif else return, results
end

function getfloodfillimage, grayimage, xx, yy, floodthreshold
  thresimage = grayimage le floodthreshold
  labelim = label_region(thresimage)
  thislabel = labelim[xx,yy]
  results = grayimage*0
  wherefill = array_indices(results, where(labelim eq thislabel, numpix))
  ;print, thislabel,max(labelim),numpix, floodthreshold
  results[wherefill[0,*],wherefill[1,*]] = 1
  return, thresimage
end

pro sfbdrawevents, event

  common sfa, faimage, grayimage, binaryimage, background, iscancel, sfawindow, oldwindow, stackmode
  common interactive, imode, drawwindow, undoimage
  common famask, famaskbinary
  common cellmask, cellmaskbinary, distancemap, averageimage, cellmaskgrayscale, cellmaskdynamic
  ;common node_dataset, properties, image_stack, node_table,node_searchbuffer,node_table_array
  ;common dataset, properties, stackraw, stackproc, auxilliary, driftparameters, shiftparameters
  Common meshwork_dataset, properties, image_set, filamentmask, nodemask, node_table, pore_table, filament_table, angle_table


  if size(properties,/type) ne 8 then begin
    help, properties
    return
  endif

  wxsz = 1024. & wysz = 1024.
  width = (size(binaryimage,/dimensions))[0]
  height = (size(binaryimage,/dimensions))[1]
  mgw = wxsz/width
  drawwindow = !d.window

  properties.xpixels = width
  properties.ypixels = height
  ;help, mgw
  ; help, properties
  ;help, event
  ;help, imode
  ;help,event.type

  if imode eq 0 then return
  IF event.type GT 2 THEN RETURN

  eventtypes = ['NONE','RC','LC','MC']
  if (event.type eq 0) and (event.press eq 1) then thisevent = eventtypes[1] else $
    if (event.type eq 0) and (event.press eq 4) then thisevent = eventtypes[2] else $
    if (event.type eq 0) and (event.press eq 2) then thisevent = eventtypes[3] else $
    thisevent =  eventtypes[0]

  dostack = widget_info(widget_info(event.top,find_by_uname='WID_DROPLIST_STACK'),/droplist_select)

  ;help, thisevent
  ; print, event.x, event.y
  case thisevent of
    'RC': begin
      print, 'RC'
      datax = event.x/mgw
      datay = event.y/mgw
      print,' here:',datax, datay
      if fix(event.x) eq 0 then datax = 0.
      if fix(event.y) eq 0 then datay = 0.
      if datax ge properties.xpixels-2 then datax = properties.xpixels-1
      if datay ge properties.ypixels-2 then datay = properties.ypixels-1
      print,' here:',datax, datay

      Widget_Control, event.top, Get_UValue=info, /No_Copy

      if ~dostack then case imode-1 of
      2: binaryimage = binaryimage or getfloodfillimage(grayimage, datax, datay, grayimage[datax,datay])
      3: binaryimage = binaryimage and ~getfloodfillimage(grayimage, datax, datay, grayimage[datax,datay])
      4:  binaryimage = binaryimage or ~getfloodfillimage(grayimage, datax, datay, grayimage[datax,datay])
      5: binaryimage = binaryimage xor getselectedregionimage(binaryimage, datax, datay)
      6:binaryimage =  binaryimage or getselectedregionimage(binaryimage, datax, datay,/grow)
      7: binaryimage = combinebinary(binaryimage,getselectedregionimage(binaryimage, datax, datay,/shrink),update=3)
      8:  binaryimage = combinebinary(binaryimage,getselectedregionimage(binaryimage, datax, datay,/smoothgrow),update=3)
      9:  binaryimage = combinebinary(binaryimage,getselectedregionimage(binaryimage, datax, datay,/smoothshrink),update=3)
      10: binaryimage=combinebinary(binaryimage,splitregions(getselectedregionimage(binaryimage, datax, datay),grayimage,/keepmax),update= 3)
      else:
    endcase

    if dostack then case imode-1 of
    2: stackprocess, event, datax, datay, imodex = imode-1, threshold = grayimage[datax,datay]
    3: stackprocess, event, datax, datay, imodex = imode-1, threshold = grayimage[datax,datay]
    4: stackprocess, event, datax, datay, imodex = imode-1, threshold = grayimage[datax,datay]
    5: stackprocess, event, datax, datay, imodex = imode-1
    6: stackprocess, event, datax, datay, imodex = imode-1
    7: stackprocess, event, datax, datay, imodex = imode-1
    8: stackprocess, event, datax, datay, imodex = imode-1
    9: stackprocess, event, datax, datay, imodex = imode-1
    else:
  endcase

  showgrayoverlay, event
  ;help, imode
  if (imode ge 1) and (imode le 3) then begin
    if info.npnts eq 0 then begin
      info.xpnts[0] = datax
      info.ypnts[0] = datay
      info.npnts = 1
    endif else begin
      info.xpnts[info.npnts] = datax
      info.ypnts[info.npnts] = datay
      info.npnts = info.npnts+1
    endelse
    linx = info.xpnts[0:info.npnts-1]
    liny = info.ypnts[0:info.npnts-1]
    ;help, linx
    ;help, info
    plots, linx*mgw, liny*mgw,/device,color=cgcolor('red')
  endif

  widget_control, event.top, set_uvalue=info,/no_copy
end
'LC': begin
  print, 'LC'
  datax = event.x/mgw
  datay = event.y/mgw
  if fix(event.x) eq 0 then datax = 0.
  if fix(event.y) eq 0 then datay = 0.
  if datax ge properties.xpixels-2 then datax = properties.xpixels-1
  if datay ge properties.ypixels-2 then datay = properties.ypixels-1
  ;print, 'LC',event.x/mgw, event.y/mgw
  Widget_Control, event.top, Get_UValue=info, /No_Copy

  help, info
  help, imode
  if (imode ge 1) and (imode le 3) then begin

    if info.npnts gt 0 then begin
      info.xpnts[info.npnts-1] =0
      info.ypnts[info.npnts-1] =0
      info.npnts = info.npnts-1
    endif

    if info.npnts gt 0 then begin
      linx = info.xpnts[0:info.npnts-1]
      liny = info.ypnts[0:info.npnts-1]
      if (imode ge 1) and (imode le 3) then begin
        showgrayoverlay, event
        plots, linx*mgw, liny*mgw,/device,color=cgcolor('red')
      endif
    endif

  endif

  widget_control, event.top, set_uvalue=info,/no_copy
end
'MC': begin
  datax = event.x/mgw
  datay = event.y/mgw
  if fix(event.x) eq 0 then datax = 0.
  if fix(event.y) eq 0 then datay = 0.
  if datax ge properties.xpixels-2 then datax = properties.xpixels-1
  if datay ge properties.ypixels-2 then datay = properties.ypixels-1

  ; print, 'MC',event.x/mgw, event.y/mgw
  Widget_Control, event.top, Get_UValue=info, /No_Copy

  if (imode ge 1) and (imode le 3) then begin
    if info.npnts gt 2 then begin
      linx = info.xpnts[0:info.npnts-1]
      liny = info.ypnts[0:info.npnts-1]
      ;plots, [linx,linx[0]]*mgw, [liny,liny[0]]*mgw,/device,color=cgcolor('yellow')
      case imode of
        1: begin ; add
          binaryimage[polyfillv(linx, liny, width, height)] = 1
          if dostack then begin
            imreg = binaryimage*0
            imreg[polyfillv(linx, liny, width, height)] = 1
            stackprocess, event, datax, datay, edit = imreg
          endif
          showgrayoverlay, event
          plots, [linx,linx[0]]*mgw, [liny,liny[0]]*mgw,/device,color=cgcolor('yellow')
        end
        2: begin ; subtract
          binaryimage[polyfillv(linx, liny, width, height)] = 0
          if dostack then  begin
            imreg = binaryimage*0
            imreg[polyfillv(linx, liny, width, height)] = 1
            stackprocess, event, datax, datay, edit = imreg,/subtract
          endif
          showgrayoverlay, event
          plots, [linx,linx[0]]*mgw, [liny,liny[0]]*mgw,/device,color=cgcolor('yellow')
        end
        else:
      endcase
      info.npnts = 0
      info.xpnts = info.xpnts*0
      info.ypnts = info.ypnts*0
    endif
  endif
  widget_control, event.top, set_uvalue=info,/no_copy
end
else:
endcase
end

pro showgrayoverlay, event, bw =bw, widget = widget , nooverlay = nooverlay, average=average
  common sfa, faimage, grayimage, binaryimage, background, iscancel, sfawindow, oldwindow, stackmode
  common interactive, imode, drawwindow, undoimage

  wxsz = 1024 & wysz = 1024
  if size(grayimage,/n_dimensions) ge 2 then  im = grayimage[*,*,0] else return
  if keyword_set(average) then im = average
  imscl = congrid(im, wxsz, wysz,/interp)

  if keyword_set(widget) then colormode = widget_info(Widget_Info(widget, FIND_BY_UNAME='WID_DROPLIST_COLOR'),/droplist_select) else $
    colormode = widget_info(Widget_Info(event.top, FIND_BY_UNAME='WID_DROPLIST_COLOR'),/droplist_select)
  ctx = [0,13,5,25,3,33,34,0]
  ct = ctx[colormode]
  ;print, ct
  cgloadct, ct
  if colormode eq 7 then cgloadct, ct, /reverse

  device, decompose = 0
  cgimage, bytarr(wxsz,wysz)
  tvscl, imscl

  if ~keyword_set(nooverlay) then begin
    scaledmask =  congrid(bytscl(~binaryimage, min =0, max = 1),wxsz, wysz)
    xt = keyword_set(bw) ? 0:13
    cgImage, scaledmask,ctIndex = xt, transparent = 70,position = [0,0,1,1]
  endif

end



pro WID_BASE_sfb_event, Event
  common sfa, faimage, grayimage, binaryimage, background, iscancel, sfawindow, oldwindow, stackmode
  common interactive, imode, drawwindow, undoimage
  common dataset, properties, stackraw, stackproc, auxilliary, driftparameters, shiftparameters
  common famask, famaskbinary
  common cellmask, cellmaskbinary, distancemap, averageimage, cellmaskgrayscale, cellmaskdynamic


  wTarget = (widget_info(Event.id,/NAME) eq 'TREE' ?  $
    widget_info(Event.id, /tree_root) : event.id)
  wWidget =  Event.top

  if stackmode ne 0 then widget_control,widget_info(wwidget,find_by_uname = 'WID_SLIDER_FRAME'),get_value = thisframe else thisframe = -1
  ;if size(properties,/type) eq 0 then return


  case wTarget of
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_THRES'): begin
      binarize, event
      showbinaryimage,event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_VIEWMASK'):  showbinaryimage, event
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_VIEWREF'):    showgrayoverlay,event,/nooverlay
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_VIEWOVERLAY'): showgrayoverlay, event
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_MORPH'): begin
      morphological, event
      showbinaryimage, event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_SIZE'): begin
      filterbysize, event
      showbinaryimage, event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_DONE'): begin

      iscancel = 0
      widget_control, wwidget,/destroy
      wset, oldwindow
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_CANCEL'): begin
      iscancel = 1
      widget_control, wwidget,/destroy
      wset, oldwindow
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_DRAW_MAIN'):          sfbdrawevents, event
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_IMODEUNDO'):     undodrawbin, event
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_IMODESTOP'): begin
      imode = 0
      widget_control,widget_info(event.top,find_by_uname='WID_DROPLIST_IMODE'),set_droplist_select=0
      showbinaryimage, event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_IMODESTART'): imode = widget_info(widget_info(event.top,find_by_uname='WID_DROPLIST_IMODE'),/droplist_select)

    Widget_Info(wWidget, FIND_BY_UNAME='WID_DROPLIST_IMODE'): begin
      if( Tag_Names(Event, /STRUCTURE_NAME) eq 'WIDGET_DROPLIST' )then $
        imode = widget_info(widget_info(event.top,find_by_uname='WID_DROPLIST_IMODE'),/droplist_select)
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_DROPLIST_COLOR'): begin
      colormode = widget_info(Widget_Info(wwidget, FIND_BY_UNAME='WID_DROPLIST_COLOR'),/droplist_select)
      ctx = [0,13,5,25,3,33,34,0]
      ct = ctx[colormode]
      device, decompose=0
      cgloadct, ct
      if colormode eq 7 then cgloadct, ct, /reverse
      ;print, ct
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_SIZES'): begin
      filterbysize, event,/small
      showbinaryimage, event
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_FILL'): begin
      filterbysize, event,/fill
      showbinaryimage, event
    end

    Widget_Info(wWidget, FIND_BY_UNAME='WID_DROPLIST_GRAY'): begin
      grayimage = widget_info(Widget_Info(wWidget, FIND_BY_UNAME='WID_DROPLIST_GRAY'),/droplist_select)? mean(stackproc,dimension = 3) :stackproc[*,*,thisframe]
      showfaimage, event
    end

    Widget_Info(wWidget, FIND_BY_UNAME='WID_SLIDER_FRAME'): begin
      grayoption = widget_info(Widget_Info(wWidget, FIND_BY_UNAME='WID_DROPLIST_GRAY'),/droplist_select)
      case stackmode of
        0:
        1: begin
          if size(stackproc, /type) eq 0 then return
          faimage = stackproc[*,*,event.value]
          grayimage = grayoption? mean(stackproc,dimension = 3):stackproc[*,*,event.value]
          if size(famaskbinary, /type) eq 0 then return
          binaryimage = famaskbinary[*,*,event.value]
          showgrayoverlay,event
          ;print, 'here'
        end
        2: begin
          if size(stackproc, /type) eq 0 then return
          faimage = stackproc[*,*,event.value]
          grayimage = grayoption? mean(stackproc,dimension = 3): stackproc[*,*,event.value]
          if size(cellmaskdynamic, /type) eq 0 then return
          binaryimage = cellmaskdynamic[*,*,event.value]
          showgrayoverlay,event
        end
      endcase
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_COMMITPROCEED'): begin
      if size(famaskbinary,/type) eq 0 then return
      if size(binaryimage, /type) eq 0 then return
      grayoption = widget_info(Widget_Info(wWidget, FIND_BY_UNAME='WID_DROPLIST_GRAY'),/droplist_select)
      case stackmode of
        0: begin
        end
        1: begin
          if size(stackproc, /type) eq 0 then return
          if size(famaskbinary, /type) eq 0 then return
          famaskbinary[*,*,thisframe] = binaryimage
          thisframe = thisframe+1 <(properties.frames-1)
          widget_control,widget_info(wwidget, find_by_uname='WID_SLIDER_FRAME'),set_value=thisframe
          faimage = stackproc[*,*,thisframe]
          grayimage = grayoption? mean(stackproc,dimension = 3): stackproc[*,*,thisframe]
          binaryimage = famaskbinary[*,*,thisframe]
          showgrayoverlay,event
        end
        2: begin
          if size(cellmaskdynamic, /type) eq 0 then return
          if size(stackproc, /type) eq 0 then return
          cellmaskdynamic[*,*,thisframe] = binaryimage
          thisframe = thisframe+1 <(properties.frames-1)
          widget_control,widget_info(wwidget, find_by_uname='WID_SLIDER_FRAME'),set_value=thisframe
          faimage = stackproc[*,*,thisframe]
          grayimage = grayoption? mean(stackproc,dimension = 3): stackproc[*,*,thisframe]
          binaryimage = cellmaskdynamic[*,*,thisframe]
          showgrayoverlay,event
        end
      endcase
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_COMMIT'): begin
      if size(binaryimage, /type) eq 0 then return
      case stackmode of
        0: begin
          undoimage = binaryimage
          print, 'here'
        end
        1: begin
          if size(stackproc, /type) eq 0 then return
          if size(famaskbinary, /type) eq 0 then return
          famaskbinary[*,*,thisframe] = binaryimage
        end
        2: begin
          if size(cellmaskdynamic, /type) eq 0 then return
          if size(stackproc, /type) eq 0 then return
          cellmaskdynamic[*,*,thisframe] = binaryimage
        end
      endcase
    end
    Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_COMPAREFAMASK'): case stackmode of
    0: begin
    end
    1: begin
      if size(famaskbinary,/type) eq 0 then return
      if size(binaryimage, /type) eq 0 then return
      showfaimage, event
      oldbin = famaskbinary[*,*,thisframe]
      plotboundary,oldbin, color='green',/boundary
      plotboundary, binaryimage, color='red',/boundary
    end
    2: begin
      if size(cellmaskdynamic,/type) eq 0 then return
      if size(binaryimage, /type) eq 0 then return
      showfaimage, event
      oldbin = cellmaskdynamic[*,*,thisframe]
      plotboundary,oldbin, color='green',/boundary
      plotboundary, binaryimage, color='red',/boundary
    end
  endcase
  Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_COMPAREFRAMES'): case stackmode of
  0: begin
  end
  1: begin
    if size(famaskbinary,/type) eq 0 then return
    if size(binaryimage, /type) eq 0 then return
    showfaimage, event
    oldbin = famaskbinary[*,*,(thisframe-1)>0]
    nextbin = famaskbinary[*,*,(thisframe+1)<(properties.frames-1)]
    plotboundary,oldbin, color='green',/boundary
    plotboundary, binaryimage, color='yellow',/boundary
    plotboundary, nextbin, color = 'red',/boundary
  end
  2: begin
    if size(cellmaskdynamic,/type) eq 0 then return
    if size(binaryimage, /type) eq 0 then return
    showfaimage, event
    oldbin = cellmaskdynamic[*,*,(thisframe-1)>0]
    nextbin = cellmaskdynamic[*,*,(thisframe+1)<(properties.frames-1)]
    plotboundary,oldbin, color='green',/boundary
    plotboundary, binaryimage, color='yellow',/boundary
    plotboundary, nextbin, color = 'red',/boundary
    ; print, thisframe, (thisframe-1)>0,(thisframe+1)<(properties.frames-1),properties.frames
  end
endcase
;        Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_CUSTOMBINARY'):  begin
;          custommask = binaryimage*0.
;          results = morphometry(binaryimage, grayscale=faimage, custommask=custommask, $
;            fieldcode= widget_info(widget_info(event.top,find_by_uname='WID_DROPLIST_MASK'),/droplist_select))
;          if ~widget_info(widget_info(event.top,find_by_uname='WID_DROPLIST_PLOTMODE'),/droplist_select) then cgimage, bytarr(1024,1024)
;          ;help, custommask
;          print, min(custommask),max(custommask)
;          displaybin, event, binary=custommask ,transparent =widget_info(widget_info(event.top,find_by_uname='WID_DROPLIST_PLOTMODE'),/droplist_select)
;        end
;        Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_HISTOGRAM'): begin
;          results = morphometry(binaryimage, grayscale=faimage,  $
;            fieldcode= widget_info(widget_info(event.top,find_by_uname='WID_DROPLIST_MASK'),/droplist_select))
;          customhist, event, stat=results,fieldcode=widget_info(widget_info(event.top,find_by_uname='WID_DROPLIST_MASK'),/droplist_select),$
;            noerase=widget_info(widget_info(event.top,find_by_uname='WID_DROPLIST_PLOTMODE'),/droplist_select)
;        end
;         Widget_Info(wWidget, FIND_BY_UNAME='WID_BUTTON_PROP'):begin
;           custommask = binaryimage*0.
;           results = morphometry(binaryimage, grayscale=faimage, custommask=custommask, $
;           fieldcode= widget_info(widget_info(event.top,find_by_uname='WID_DROPLIST_MASK'),/droplist_select))
;           widget_control,widget_info(event.top,find_by_uname='WID_SLIDER_MINV'),get_value = minv
;           widget_control,widget_info(event.top,find_by_uname='WID_SLIDER_MAXV'),get_value = maxv
;           binaryimage = filtermorphometry(custommask, min=minv, max=maxv)
;           showbinaryimage, event
;
;          end
else:
endcase
end

pro undodrawbin, event
  common sfa, faimage, grayimage, binaryimage, background, iscancel, sfawindow, oldwindow, stackmode
  common interactive, imode, drawwindow, undoimage

  if size(undoimage,/type ) eq 0 then return
  binaryimage = undoimage
  ;showbinaryimage, event
  showgrayoverlay, event
end

pro initializesfb, wWidget
  common sfa, faimage, grayimage, binaryimage, background, iscancel, sfawindow, oldwindow, stackmode
  common interactive, imode, drawwindow, undoimage
  common mouse, mousedown, infoinitial, mousemode

  imode = 0
  wxsz = 1024 & wysz = 1024
  ;  if size(binaryimage,/n_dimensions) ge 2 then  im = binaryimage[*,*,0] else return
  ;  imscl = congrid(im, wxsz, wysz,/interp)
  ;  loadct, 3
  ;  device, decompose = 0
  ;  tvscl, imscl
  drawwindow = !d.window
  undoimage = binaryimage


  info = {image: grayimage, $
    wid:drawwindow, $
    drawID: widget_info(wWidget,find_by_uname='WID_DRAW_MAIN'), $
    pixID:-1, $
    xsize:wxsz, $
    ysize:wysz, $
    sx:-1,$
    sy:-1,$
    imode:imode, $
    xpnts:dblarr(1000),$
    ypnts:dblarr(1000),$
    npnts:0L,$
    binary:binaryimage}
  infoinitial = info
  widget_control, wWidget, set_uvalue =info, /no_copy

  showgrayoverlay,event, /bw , widget=wwidget
end

pro quit_WID_BASE_SFB, wWidget
  common sfa, faimage, grayimage, binaryimage, background, iscancel, sfawindow, oldwindow, stackmode
  common dataset, properties, stackraw, stackproc, auxilliary, driftparameters, shiftparameters
  common famask, famaskbinary
  common cellmask, cellmaskbinary, distancemap, averageimage, cellmaskgrayscale, cellmaskdynamic
  common script, recordmode, scripttable, numberentry
  common interactive, imode, drawwindow, undoimage

  ;iscancel = 1
  wset, oldwindow

end

pro WID_BASE_sfb, GROUP_LEADER=wGroup, _EXTRA=_VWBExtra_,stack=stack
  common sfa, faimage, grayimage, binaryimage, background, iscancel, sfawindow, oldwindow, stackmode
  common dataset, properties, stackraw, stackproc, auxilliary, driftparameters, shiftparameters

  if keyword_set(stack) then begin
    case stack of
      0: tt = 'Binary Image Editor (c) 2013-2014 Kanchanawong Lab, MBI/NUS'
      1: tt = 'FA mask Editor (c) 2013-2014 Kanchanawong Lab, MBI/NUS'
      2: tt = 'Cell mask Editor (c) 2013-2014 Kanchanawong Lab, MBI/NUS'
    endcase
  endif else tt =   'Binary Image Editor (c) 2013-2018 Kanchanawong Lab, MBI/NUS'


  WID_BASE_sfb = Widget_Base( GROUP_LEADER=wGroup,  $
    UNAME='WID_BASE_sfb' , XOFFSET=5 ,YOFFSET=5  $
    ,SCR_XSIZE=1300 ,SCR_YSIZE=1064,XSIZE=1300,YSIZE=1064,/modal,notify_realize='initializesfb' $
    , TITLE=tt  ,SPACE=3 ,XPAD=3 ,YPAD=3 ,kill_notify='quit_WID_BASE_SFB')

  WID_DRAW_main = Widget_Draw(WID_BASE_sfb,  $
    UNAME='WID_DRAW_MAIN' ,XOFFSET=210+50 ,YOFFSET=5  $
    ,SCR_XSIZE=1024 ,SCR_YSIZE=1024  $
    ,/BUTTON_EVENTS)

  WID_BUTTON_commit = Widget_Button(wid_base_sfb,  $
    UNAME='WID_BUTTON_COMMIT' ,XOFFSET=10 ,YOFFSET=360+35  $
    ,SCR_XSIZE=50 ,SCR_YSIZE=30 ,/ALIGN_CENTER ,VALUE='Commit')
  if keyword_set(stack) then begin
    WID_slider_frame =   Widget_Slider(WID_BASE_sfb,  $
      UNAME='WID_SLIDER_FRAME' ,XOFFSET=215 ,YOFFSET=5  $
      ,SCR_XSIZE=45 ,SCR_YSIZE=1000 ,TITLE='',MAXIMUM=properties.frames-1,value = 0.,/vertical)
    WID_BUTTON_overlayrevert = Widget_Button(wid_base_sfb,  $
      UNAME='WID_BUTTON_COMPAREFAMASK' ,XOFFSET=10 ,YOFFSET=360  $
      ,SCR_XSIZE=180 ,SCR_YSIZE=30 ,/ALIGN_CENTER ,VALUE='Compare old(G) vs. new mask(R)')
    WID_BUTTON_overlayframes = Widget_Button(wid_base_sfb,  $
      UNAME='WID_BUTTON_COMPAREFRAMES' ,XOFFSET=10 ,YOFFSET=360-35  $
      ,SCR_XSIZE=180 ,SCR_YSIZE=30 ,/ALIGN_CENTER ,VALUE='prev-G/current-Y/next-R')

    WID_BUTTON_commit = Widget_Button(wid_base_sfb,  $
      UNAME='WID_BUTTON_COMMITPROCEED' ,XOFFSET=65 ,YOFFSET=360+35  $
      ,SCR_XSIZE=125 ,SCR_YSIZE=30 ,/ALIGN_CENTER ,VALUE='Commit and Proceed')
  endif


  ;  WID_DROPLIST_CT = Widget_Droplist(WID_BASE_sfb,  $
  ;    UNAME='WID_DROPLIST_CT' ,XOFFSET=10,YOFFSET=500-35 ,SCR_XSIZE=180  $
  ;    ,SCR_YSIZE=35 ,VALUE=['Grayscale','Red Temperature'],title='Ref. Image Color')

  WID_BUTTON_viewref = Widget_Button(WID_BASE_sfb,  $
    UNAME='WID_BUTTON_VIEWREF' ,XOFFSET=10 ,YOFFSET=500-35*2  $
    ,SCR_XSIZE=180 ,SCR_YSIZE=30 ,/ALIGN_CENTER ,VALUE='View Reference Image');

  WID_DROPLIST_grayimage = Widget_Droplist(WID_BASE_sfb,  $
    UNAME='WID_DROPLIST_GRAY' ,XOFFSET=10,YOFFSET=500-35 ,SCR_XSIZE=180  $
    ,SCR_YSIZE=26 ,VALUE=['Use current frame grayscale','Stack averaged grayscale'])

  WID_BUTTON_viewmask = Widget_Button(WID_BASE_sfb,  $
    UNAME='WID_BUTTON_VIEWMASK' ,XOFFSET=10 ,YOFFSET=500  $
    ,SCR_XSIZE=180 ,SCR_YSIZE=30 ,/ALIGN_CENTER ,VALUE='View Binary Mask');

  WID_BUTTON_viewoverlay = Widget_Button(WID_BASE_sfb,  $
    UNAME='WID_BUTTON_VIEWOVERLAY' ,XOFFSET=10 ,YOFFSET=500+35  $
    ,SCR_XSIZE=180 ,SCR_YSIZE=30 ,/ALIGN_CENTER ,VALUE='View Mask+Grayscale');

  WID_DROPLIST_MORPH = Widget_Droplist(WID_BASE_sfb,  $
    UNAME='WID_DROPLIST_MORPH' ,XOFFSET=10,YOFFSET=500+35+50 ,SCR_XSIZE=180  $
    ,SCR_YSIZE=35 ,VALUE=['Grow (Dilate)','Shrink (Erode)','Smooth&grow (Close)','Smooth&shrink (Open)','Thin'])
  WID_BUTTON_morph = Widget_Button(WID_BASE_sfb,  $
    UNAME='WID_BUTTON_MORPH' ,XOFFSET=10 ,YOFFSET=500+35+50+35  $
    ,SCR_XSIZE=180 ,SCR_YSIZE=30 ,/ALIGN_CENTER ,VALUE='Morphological Operation on Mask')

  wid_base_min= widget_base(WID_BASE_sfb, xoffset=10, yoffset = 700, scr_xsize = 190, scr_ysize=80)
  wid_slider_min = cw_fslider(wid_base_min, /DOUBLE,minimum=0, maximum=2000.,scroll = 1., $
    title='Minimum Region Size (pixels)',value = 0.,xsize=180,ysize = 80,/edit, uname='WID_SLIDER_MIN',format='(I7)')
  wid_base_max= widget_base(WID_BASE_sfb, xoffset=10, yoffset = 700+80, scr_xsize = 190, scr_ysize=80)
  wid_slider_max = cw_fslider(wid_base_max, /DOUBLE,minimum=0, maximum=2000.,scroll = 1., $
    title='Maximum Region Size (pixels)',value = 1000.,xsize=180,ysize = 80,/edit, uname='WID_SLIDER_MAX',format='(I7)')
  WID_BUTTON_size = Widget_Button(WID_BASE_sfb,  $
    UNAME='WID_BUTTON_SIZE' ,XOFFSET=10 ,YOFFSET=700+80+80  $
    ,SCR_XSIZE=180 ,SCR_YSIZE=30 ,/ALIGN_CENTER ,VALUE='Perform Size Exclusion')


  WID_BUTTON_done = Widget_Button(WID_BASE_sfb,  $
    UNAME='WID_BUTTON_DONE' ,XOFFSET=10 ,YOFFSET=960+5  $
    ,SCR_XSIZE=180 ,SCR_YSIZE=30 ,/ALIGN_CENTER ,VALUE='DONE')
  WID_BUTTON_cancel = Widget_Button(WID_BASE_sfb,  $
    UNAME='WID_BUTTON_CANCEL' ,XOFFSET=10 ,YOFFSET=960+5+35  $
    ,SCR_XSIZE=180 ,SCR_YSIZE=30 ,/ALIGN_CENTER ,VALUE='CANCEL')

  WID_BUTTON_imodestart = Widget_Button(WID_BASE_sfb,  $
    UNAME='WID_BUTTON_IMODESTART' ,XOFFSET=10 ,YOFFSET=10  $
    ,SCR_XSIZE=180 ,SCR_YSIZE=30 ,/ALIGN_CENTER ,VALUE='Start/Reset Interactive Mode')
  WID_DROPLIST_IMODE = Widget_Droplist(WID_BASE_sfb,  $
    UNAME='WID_DROPLIST_IMODE' ,XOFFSET=10,YOFFSET=10+35*1+5 ,SCR_XSIZE=180  $
    ,SCR_YSIZE=25 ,VALUE=['None','Draw Area to Add','Draw Area to Remove', 'Fill flood add to selection','Fill flood subtraction','Inverse fill flood selection','Remove selected region',$
    'Grow selected region','Shrink selected region','Smooth&grow selected region','Smooth&shrink selected region','Refine by k-Means'])
  WID_DROPLIST_STACK = Widget_Droplist(WID_BASE_sfb,  $
    UNAME='WID_DROPLIST_STACK' ,XOFFSET=10,YOFFSET=10+35*1+5+30 ,SCR_XSIZE=180  $
    ,SCR_YSIZE=26 ,VALUE=['Current frame','Whole stack'])

  WID_BUTTON_imodeundo = Widget_Button(WID_BASE_sfb,  $
    UNAME='WID_BUTTON_IMODEUNDO' ,XOFFSET=10 ,YOFFSET=10+35*3  $
    ,SCR_XSIZE=180 ,SCR_YSIZE=30 ,/ALIGN_CENTER ,VALUE='Undo Editing')
  WID_BUTTON_imodestop = Widget_Button(WID_BASE_sfb,  $
    UNAME='WID_BUTTON_IMODESTOP' ,XOFFSET=10 ,YOFFSET=10+35*4  $
    ,SCR_XSIZE=180 ,SCR_YSIZE=30 ,/ALIGN_CENTER ,VALUE='Finish Interactive Mode')


  WID_DROPLIST_color = Widget_Droplist(WID_BASE_sfb,  $
    UNAME='WID_DROPLIST_COLOR' ,XOFFSET=5 ,YOFFSET=435-75+60-150 ,SCR_XSIZE=200  $
    ,SCR_YSIZE=25 ,TITLE='Color Scale' ,VALUE=['Grayscale #0', 'Rainbow','#5 Std Gamma II', '#25 Mac Style','Red Temperature # 3','#33 Blue-Red','#34 Rainbow','Inverse Grayscale'])

  WID_BUTTON_sizes = Widget_Button(WID_BASE_sfb,  $
    UNAME='WID_BUTTON_SIZES' ,XOFFSET=10 ,YOFFSET=700+80+80+35  $
    ,SCR_XSIZE=180 ,SCR_YSIZE=30 ,/ALIGN_CENTER ,VALUE='Exclude Small Regions only')
  WID_BUTTON_fill = Widget_Button(WID_BASE_sfb,  $
    UNAME='WID_BUTTON_FILL' ,XOFFSET=10 ,YOFFSET=700+80+80+35+35  $
    ,SCR_XSIZE=180 ,SCR_YSIZE=30 ,/ALIGN_CENTER ,VALUE='Fill Small Holes (use min slider)')

  ;  WID_BUTTON_custmask = Widget_Button(WID_BASE_sfb,  $
  ;    UNAME='WID_BUTTON_CUSTOMBINARY' ,XOFFSET=1300+15 ,YOFFSET=120+35*12  $
  ;    ,SCR_XSIZE=170 ,SCR_YSIZE=30 ,/ALIGN_CENTER ,VALUE='Custom Binary Mask')
  ;  WID_BUTTON_custhist = Widget_Button(WID_BASE_sfb,  $
  ;    UNAME='WID_BUTTON_HISTOGRAM' ,XOFFSET=1300+15 ,YOFFSET=120+35*13  $
  ;    ,SCR_XSIZE=170 ,SCR_YSIZE=30 ,/ALIGN_CENTER ,VALUE='Histogram')
  ;  WID_DROPLIST_mask = Widget_Droplist(WID_BASE_sfb,  $
  ;    UNAME='WID_DROPLIST_MASK' ,XOFFSET=1300+5 ,YOFFSET=120+35*14 ,SCR_XSIZE=200  $
  ;    ,SCR_YSIZE=25 ,TITLE='Properties' ,VALUE=['Area','Axial Ratio','Density','Length','Total Intensity','Perimeter','Dist. to Edge (COM)','Dist to Edge (Avg)','Orientation'])
  ;  WID_DROPLIST_plotmode = Widget_Droplist(WID_BASE_sfb,  $
  ;    UNAME='WID_DROPLIST_PLOTMODE' ,XOFFSET=1300+5 ,YOFFSET=120+35*15 ,SCR_XSIZE=200  $
  ;    ,SCR_YSIZE=25 ,TITLE='Option:' ,VALUE=['Overwrite','Overlay'])


  ;  wid_base_min= widget_base(WID_BASE_sfb, xoffset=10+1300, yoffset = 700, scr_xsize = 190, scr_ysize=80)
  ;  wid_slider_min = cw_fslider(wid_base_min, /DOUBLE,minimum=0, maximum=100.,scroll = 1., $
  ;    title='Minimum Value (%)',value = 0.,xsize=180,ysize = 80,/edit, uname='WID_SLIDER_MINV',format='(I7)')
  ;  wid_base_max= widget_base(WID_BASE_sfb, xoffset=10+1300, yoffset = 700+80, scr_xsize = 190, scr_ysize=80)
  ;  wid_slider_max = cw_fslider(wid_base_max, /DOUBLE,minimum=0, maximum=100.,scroll = 1., $
  ;    title='Maximum Value (%)',value = 100.,xsize=180,ysize = 80,/edit, uname='WID_SLIDER_MAXV',format='(I7)')
  ;  WID_BUTTON_size = Widget_Button(WID_BASE_sfb,  $
  ;    UNAME='WID_BUTTON_PROP' ,XOFFSET=10+1300 ,YOFFSET=700+80+80  $
  ;    ,SCR_XSIZE=180 ,SCR_YSIZE=30 ,/ALIGN_CENTER ,VALUE='Perform properties Exclusion')





  oldwindow = !d.window

  Widget_Control, /REALIZE, WID_BASE_sfb
  XManager, 'WID_BASE_SFB', WID_BASE_sfb, /NO_BLOCK

  sfawindow = !d.window

end
