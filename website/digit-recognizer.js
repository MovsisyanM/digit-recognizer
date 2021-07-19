holder = document.getElementById("canvasHolder");
canvas = document.getElementById("inputCanvas");
ctx = canvas.getContext("2d");

let painting = false;
let maxX = 0;
let minX = 0;
let maxY = 0;
let minY = 0;
let pathsX = [];
let pathsY = [];



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


// Called when mousedown
function startDraw(e) {
    // Reset canvas
    ctx.fillStyle = "#ffffff";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = "#000000";

    pathsX = [];
    pathsY = [];


    // Init brush
    painting = true
    ctx.lineCap = "round";
    ctx.lineWidth = canvas.height/10;


    // Getting current x, y coordinates relative to canvas top left
    let currX = e.pageX - canvas.offsetLeft;
    let currY = e.pageY - holder.offsetTop;

    maxX = currX;
    minX = currX;
    maxY = currY;
    minY = currY;

    // Start path
    ctx.moveTo(currX, currY)
    ctx.beginPath()
    draw(e)
}


// Called when mouseup
function endDraw(e) {
    painting = false
    ctx.closePath()

    let width = (maxX - minX);
    let height = (maxY - minY);

    let offsetX = canvas.width/2 - (minX + width/2);
    let offsetY = canvas.height/2 - (minY + height/2);

    // Reset canvas
    ctx.fillStyle = "#ffffff";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = "#000000";

    ctx.lineCap = "round";
    ctx.lineWidth = canvas.height/10;
    ctx.moveTo(0,0)
    ctx.beginPath()

    // For each stroke, repaint it scaled
    for (let i = 0; i < pathsX.length; i++) {
        // Drawing the line
        ctx.lineTo(pathsX[i] + offsetX, pathsY[i] + offsetY);
        ctx.stroke();
    }
    
    ctx.closePath()
}

function draw(e) {
    // If not drawing yet, idle
    if (!painting) return;

    // Setting brush size and cap
    ctx.lineCap = "round";
    ctx.lineWidth = canvas.height/10;

    // Getting current x, y coordinates relative to canvas top left
    let currX = e.pageX - canvas.offsetLeft;
    let currY = e.pageY - holder.offsetTop;

    // Drawing the line
    ctx.lineTo(currX, currY);
    ctx.stroke();

    // Set coordinate extrema
    if (currX > maxX) {
        maxX = currX
    } else if (currX < minX) {
        minX = currX
    }

    if (currY > maxY) {
        maxY = currY
    } else if (currY < minY) {
        minY = currY
    }


    // Record path
    pathsX.push(currX)
    pathsY.push(currY)
}


// Register event listeners
document.addEventListener("mouseup", endDraw);
canvas.addEventListener("mousedown", startDraw);
canvas.addEventListener("mousemove", draw);


// Registering a CTRL + Z to reset canvas
document.onkeydown = (e) => {
    if (e.key === "z" && e.ctrlKey) {
        ctx.fillStyle = "#ffffff";
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        ctx.fillStyle = "#000000";
    }
}

correctHeight()