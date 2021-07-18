holder = document.getElementById("canvasHolder");
canvas = document.getElementById("inputCanvas");
ctx = canvas.getContext("2d");

let painting = false;

ctx.lineWidth = 13;
ctx.lineCap = "round";
ctx.strokeStyle = '#000000';
ctx.fillStyle = '#000000';

function correctHeight() {
    if (holder.clientWidth < holder.clientHeight) {
        canvas.height = holder.clientWidth
        canvas.width = canvas.height
    } else {
        canvas.height = holder.clientHeight
        canvas.width = canvas.height
    }
}

function startDraw(e) {
    painting = true
    ctx.lineCap = "round";
    ctx.lineWidth = 13;
    ctx.moveTo(e.pageX - holder.offsetLeft, e.pageY - holder.offsetTop)
    ctx.beginPath()
    draw(e)
}

function endDraw(e) {
    painting = false
    ctx.closePath()
}

function draw(e) {
    if (!painting) return;

    ctx.lineCap = "round";
    ctx.lineWidth = 13;
    ctx.lineTo(e.pageX - canvas.offsetLeft, e.pageY - holder.offsetTop);
    ctx.stroke();
}

canvas.addEventListener("mouseup", endDraw);
canvas.addEventListener("mousedown", startDraw);
canvas.addEventListener("mousemove", draw);

document.onkeydown = (e) => {
    if (e.key === "z" && e.ctrlKey) {
        ctx.fillStyle = "#ffffff";
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        ctx.fillStyle = "#000000";
    }
    console.log(e.key)
}

correctHeight()