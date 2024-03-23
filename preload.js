const { contextBridge, ipcRenderer } = require('electron/renderer')

contextBridge.exposeInMainWorld('versions', {
  node: () => process.versions.node,
  chrome: () => process.versions.chrome,
  electron: () => process.versions.electron
})

contextBridge.exposeInMainWorld('runpython', {
    send: async (data) => await ipcRenderer.invoke('python_detect', data),
    analyze: async (data) => await ipcRenderer.invoke('python_analyze', data),
    on: (channel, func) => {
        ipcRenderer.once(channel, (event,data)=>func(data))
    }
})
