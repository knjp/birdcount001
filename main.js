const { app, BrowserWindow , ipcMain} = require('electron/main')
const path = require('node:path')
const { PythonShell } = require('python-shell')

const createWindow = () => {
  const win = new BrowserWindow({
    width: 800,
    height: 600,
    webPreferences: {
        preload: path.join(__dirname, 'preload.js')
    }
  })

  win.loadFile('index.html')
  //win.webContents.openDevTools()
}

app.whenReady().then(() => {
  createWindow()

  app.on('activate', ()=>{
    if (BrowserWindow.getAllWindows().length === 0){
        createWindow()
    }
  })
})

const options = {
    mode: 'text',
    //pythonPath: __dirname,
    pythonOptions: ['-u'],
    scriptPath: __dirname + '/python'
}

ipcMain.handle('python_detect', async (event, data)=>{
    PythonShell.run('detect.py', options)
        .then((response)=>{
        event.sender.send("return_data", response);
        })
        .catch((error) =>{
            console.log(error);
        })
})

ipcMain.handle('python_analyze', async (event, data)=>{
    PythonShell.run('analyze.py', options)
        .then((response)=>{
        event.sender.send("return_data", response);
        })
        .catch((error) =>{
            console.log(error);
        })
})
