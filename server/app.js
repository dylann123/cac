const express = require("express")
const app = express()
const fs = require("fs")
const fileupload = require("express-fileupload")

const spawn = require("child_process").spawn;

require("dotenv").config()
app.use(fileupload())
app.use(express.json())

app.get("/", (req, res) => {
    res.sendFile(__dirname+"/src/main/index.html")
})

app.post("/upload", (req, res) => {
    const { image } = req.files;
    if (!image) return res.sendStatus(400);
    image.mv(__dirname + '/model/cache/' + image.name);
    res.redirect("/getresults?id="+image.name)
})

app.get("/getresults", (req, res) => {
    console.log("Get data for "+req.query["id"]);
    const pythonProcess = spawn('python', ["usemodel.py", "model/cache/" + req.query["id"]]);
    pythonProcess.stdout.on('data', (data) => {
        res.send(JSON.stringify({response: data.toString().split("\n")[0]}))
        return
    });
})

app.listen(process.env.PORT, () => {
    console.log("Server up @ " + process.env.PORT);
})
