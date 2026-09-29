"""The map the painter draws on: scroll to zoom, right-drag to pan, left-drag to paint,
hold Space to see the bare terrain.

A Streamlit custom component rather than a chart, so the view stays where you left it
while the picture underneath is redrawn after every stroke.
"""
import base64
import io

import streamlit as st

HTML = """
<div class="map">
  <canvas tabindex="-1"></canvas>
  <button type="button" title="Show the whole map">Fit</button>
</div>
"""

CSS = """
.map { position: relative; }
canvas {
  display: block; width: 100%; cursor: crosshair; touch-action: none; user-select: none;
  background: var(--st-secondary-background-color); border-radius: var(--st-base-radius);
  outline: none;
}
canvas.smart { cursor: cell; }
canvas.locked { cursor: not-allowed; }
button {
  position: absolute; top: .5rem; right: .5rem; padding: .15rem .6rem; cursor: pointer;
  font: inherit; font-size: .8rem; color: var(--st-text-color);
  background: var(--st-background-color); border: 1px solid var(--st-border-color);
  border-radius: var(--st-button-radius);
}
"""

JS = """
// Views live on the page, keyed by map, so a redraw or remount keeps your zoom.
const MAPS = (globalThis.__selenographMaps ??= new Map())

function inside(x, y, p) {
  let c = false
  for (let a = 0, b = p.length - 1; a < p.length; b = a++) {
    const [xa, ya] = p[a], [xb, yb] = p[b]
    if ((ya > y) !== (yb > y) && x < ((xb - xa) * (y - ya)) / (yb - ya) + xa) c = !c
  }
  return c
}

// Cells the stroke passes over, plus every cell whose centre it loops around.
export function selectCells(path, grid, width, height, bounds = [0, 0, 1, 1], validCells = null) {
  const [x, y, w, h] = bounds
  const ox = x * width, oy = y * height
  const cellWidth = w * width / grid, cellHeight = h * height / grid
  const allowed = validCells === null ? null : new Set(validCells)
  // Work in cell coordinates; remove roundoff at exact edges after translating
  // a clipped tile. The same stroke must select the same original painting cells.
  const snap = (v) => Math.abs(v - Math.round(v)) < 1e-9 ? Math.round(v) : v
  path = path.map(([x, y]) => [snap((x - ox) / cellWidth), snap((y - oy) / cellHeight)])
  const cw = 1, ch = 1
  const hit = new Set()
  const add = (x, y) => {
    const i = Math.floor(y / ch), j = Math.floor(x / cw)
    if (i >= 0 && j >= 0 && i < grid && j < grid) hit.add(i * grid + j)
  }
  add(...path[0])
  for (let n = 1; n < path.length; n++) {
    const [x0, y0] = path[n - 1], [x1, y1] = path[n]
    const steps = Math.ceil(Math.hypot((x1 - x0) / cw, (y1 - y0) / ch) * 3)
    for (let t = 1; t <= steps; t++) add(x0 + ((x1 - x0) * t) / steps, y0 + ((y1 - y0) * t) / steps)
  }
  if (path.length > 2) {
    const xs = path.map((p) => p[0]), ys = path.map((p) => p[1])
    const j0 = Math.max(0, Math.floor(Math.min(...xs) / cw)), j1 = Math.min(grid - 1, Math.floor(Math.max(...xs) / cw))
    const i0 = Math.max(0, Math.floor(Math.min(...ys) / ch)), i1 = Math.min(grid - 1, Math.floor(Math.max(...ys) / ch))
    for (let i = i0; i <= i1; i++)
      for (let j = j0; j <= j1; j++)
        if (inside((j + 0.5) * cw, (i + 0.5) * ch, path)) hit.add(i * grid + j)
  }
  return Array.from(hit).filter((id) => allowed === null || allowed.has(id)).sort((a, b) => a - b)
}


export default function (component) {
  const { data, parentElement, setTriggerValue } = component
  const canvas = parentElement.querySelector("canvas")
  const fitButton = parentElement.querySelector("button")
  if (!canvas || !data) return
  const ctx = canvas.getContext("2d")
  const grid = data.grid
  canvas.style.height = `${data.height}px`
  canvas.classList.toggle("smart", data.tool === "smart")
  canvas.classList.toggle("locked", data.tool === "locked")
  const viewKey = `${data.map}:${JSON.stringify(data.bounds)}`
  let s = MAPS.get(viewKey)
  if (!s) MAPS.set(viewKey, (s = { img: null, src: null, bare: null, bareSrc: null,
                                     view: null, lasso: null, pan: null, peek: false }))

  function size() {
    const r = canvas.getBoundingClientRect()
    const dpr = window.devicePixelRatio || 1
    const w = Math.max(1, Math.round(r.width * dpr)), h = Math.max(1, Math.round(r.height * dpr))
    if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h }
    return [r.width, r.height]
  }

  function fit() {
    const [w, h] = size(), iw = s.img.naturalWidth, ih = s.img.naturalHeight
    const k = Math.min(w / iw, h / ih)
    s.view = { k, fit: k, x: (w - iw * k) / 2, y: (h - ih * k) / 2 }
  }

  function draw() {
    if (!s.img) return
    const [w, h] = size()
    if (!s.view) fit()
    const v = s.view
    ctx.setTransform(canvas.width / w, 0, 0, canvas.height / h, 0, 0)
    ctx.clearRect(0, 0, w, h)
    ctx.imageSmoothingEnabled = v.k < 1          // crisp cells once zoomed in
    const pic = s.peek && s.bare ? s.bare : s.img
    ctx.drawImage(pic, v.x, v.y, s.img.naturalWidth * v.k, s.img.naturalHeight * v.k)
    if (pic === s.bare) tag(data.peek, 8, 8)
    if (data.legend) tag(data.legend, 8, h - 30)
    if (s.lasso && s.lasso.length > 1) {
      ctx.beginPath()
      s.lasso.forEach(([x, y], n) => (n ? ctx.lineTo : ctx.moveTo).call(ctx, v.x + x * v.k, v.y + y * v.k))
      ctx.lineJoin = "round"
      ctx.lineWidth = 4; ctx.strokeStyle = "rgba(0, 0, 0, 0.55)"; ctx.stroke()
      ctx.lineWidth = 2; ctx.strokeStyle = "#fff"; ctx.stroke()
    }
  }

  // A small dark label on the map.
  function tag(text, x, y) {
    ctx.font = "12px sans-serif"
    ctx.fillStyle = "rgba(0, 0, 0, 0.6)"
    ctx.fillRect(x, y, ctx.measureText(text).width + 16, 22)
    ctx.fillStyle = "#fff"
    ctx.fillText(text, x + 8, y + 15)
  }

  // Map pixel under the pointer.
  function at(e) {
    const r = canvas.getBoundingClientRect()
    return [(e.clientX - r.left - s.view.x) / s.view.k, (e.clientY - r.top - s.view.y) / s.view.k]
  }

  // Pointing at the map gives it the keyboard, so Space and Streamlit's F shortcut work
  // right after using the sidebar (a focused unit, slider or button swallows them). A
  // box you are typing in keeps the keyboard; the map dropdown only while it is open.
  const TEXT = /^(text|search|email|number|password|tel|url)$/
  const editing = (t) => !!t && (t.isContentEditable || t.tagName === "TEXTAREA" ||
                                 (t.tagName === "INPUT" && TEXT.test(t.type)))
  const busy = (a) => editing(a) &&
    !(a.getAttribute("role") === "combobox" && a.getAttribute("aria-expanded") !== "true")
  const claim = () => {
    if (canvas.getRootNode().activeElement !== canvas && !busy(document.activeElement))
      canvas.focus({ preventScroll: true })
  }
  const typing = (e) => editing(e.composedPath()[0])

  // Hold Space over the map to peek at the terrain without the unit colours.
  const peek = (on) => { if (s.peek !== on) { s.peek = on; draw() } }
  const keys = {
    keydown: (e) => {
      if (e.code !== "Space" || typing(e) || !(s.hover || s.peek)) return
      e.preventDefault()             // no page scroll, no pressing the focused widget
      peek(true)
    },
    keyup: (e) => {
      if (e.code !== "Space" || !s.peek) return
      e.preventDefault()             // a focused button would otherwise click on release
      peek(false)
    },
    blur: () => peek(false),
  }
  // Capture phase on the window, so a focused widget cannot swallow the key first.
  const listen = (set, on) => {
    for (const [type, fn] of Object.entries(set ?? {}))
      window[on ? "addEventListener" : "removeEventListener"](type, fn, true)
  }
  listen(s.keys, false)                          // the previous render's listeners
  listen((s.keys = keys), true)
  canvas.onpointerenter = () => { s.hover = true; claim() }
  canvas.onpointerleave = () => { s.hover = false }

  canvas.oncontextmenu = (e) => e.preventDefault()
  canvas.onpointerdown = (e) => {
    if (!s.view) return
    e.preventDefault()
    canvas.focus({ preventScroll: true })
    canvas.setPointerCapture(e.pointerId)
    if (e.button === 0) { s.lasso = [at(e)]; s.travel = 0; s.last = [e.clientX, e.clientY] }
    else { s.pan = [e.clientX, e.clientY]; canvas.style.cursor = "grabbing" }
  }
  canvas.onpointermove = (e) => {
    s.hover = true                               // also after a redraw, without re-entering
    claim()
    if (s.pan) {
      s.view.x += e.clientX - s.pan[0]; s.view.y += e.clientY - s.pan[1]
      s.pan = [e.clientX, e.clientY]
      draw()
    } else if (s.lasso) {
      s.travel += Math.hypot(e.clientX - s.last[0], e.clientY - s.last[1])
      s.last = [e.clientX, e.clientY]
      s.lasso.push(at(e))
      draw()
    }
  }
  canvas.onpointerup = () => {
    if (s.pan) { s.pan = null; canvas.style.cursor = "" }
    if (s.lasso) {
      const click = s.travel < 5                 // a shaky click is still a click
      const cells = selectCells(click ? s.lasso.slice(0, 1) : s.lasso, grid,
                                s.img.naturalWidth, s.img.naturalHeight, data.bounds, data.validCells)
      s.lasso = null
      draw()
      if (cells.length)
        setTriggerValue("stroke", { id: `${Date.now()}-${Math.random()}`, cells, click })
    }
  }
  canvas.onpointercancel = () => { s.pan = s.lasso = null; canvas.style.cursor = ""; draw() }
  canvas.onwheel = (e) => {
    if (!s.view) return
    e.preventDefault()
    const r = canvas.getBoundingClientRect(), mx = e.clientX - r.left, my = e.clientY - r.top
    const dy = e.deltaY * (e.deltaMode === 1 ? 16 : e.deltaMode === 2 ? 400 : 1)
    const k = Math.min(s.view.fit * 40, Math.max(s.view.fit * 0.5, s.view.k * Math.exp(-dy * 0.0015)))
    s.view.x = mx - ((mx - s.view.x) * k) / s.view.k
    s.view.y = my - ((my - s.view.y) * k) / s.view.k
    s.view.k = k
    draw()
  }
  fitButton.onclick = () => { if (s.img) { fit(); draw() } }

  if (s.bareSrc !== data.terrain) {
    const img = new Image(), src = data.terrain
    s.bareSrc = src
    img.onload = () => { if (s.bareSrc === src) { s.bare = img; draw() } }
    img.src = src
  }
  if (s.src !== data.image) {                    // keep showing the old picture until the new one loads
    const img = new Image(), src = data.image
    s.src = src
    img.onload = () => { if (s.src === src) { s.img = img; draw() } }
    img.src = src
  } else draw()

  s.observer?.disconnect()
  const observer = (s.observer = new ResizeObserver(() => draw()))
  observer.observe(canvas)
  // Tear down only what this render set up, in case a newer render is already live.
  return () => {
    observer.disconnect()
    listen(keys, false)
    if (s.keys === keys) s.keys = null
  }
}
"""

def png(image):
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def map_canvas(image, *, terrain, grid, height, key, legend="", peek="Bare terrain",
               tool="paint", bounds=(0, 0, 1, 1), valid_cells=None):
    """Show `image`, the whole map as a PIL image, for painting on a grid x grid of cells;
    `terrain` is the same map without unit colours, shown while Space is held and labelled
    `peek`. `legend`, if any, is written in the bottom corner; `tool` sets the cursor.
    Returns the last stroke as {"id", "cells": [flat cell indices], "click": bool}, or
    None. A click's cells are just the one under the pointer. `bounds` locates the
    original raster inside the displayed square; `valid_cells` excludes NoData.
    Padding never changes the stored painting grid or cell IDs."""
    # Registered on every call, not once at import: each Streamlit runtime (and each test)
    # has its own registry, and re-registering an identical component is a no-op.
    component = st.components.v2.component("selenograph_canvas", html=HTML, css=CSS, js=JS)
    result = component(key=key, data=dict(image=png(image), terrain=png(terrain), grid=grid,
                                          height=height, map=key, legend=legend, peek=peek,
                                          tool=tool, bounds=list(bounds),
                                          validCells=None if valid_cells is None else
                                          [int(i) for i, valid in enumerate(valid_cells.flat) if valid]),
                       on_stroke_change=lambda: None)
    return result.stroke
