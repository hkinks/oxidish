//! WebSocket thread: owns the socket, forwards raw binary packets to the game loop and writes
//! whatever the game loop queues. Handshake bytes are the web client's (protocol 6, key 1).

use std::io::ErrorKind;
use std::net::TcpStream;
use std::sync::mpsc::{channel, Receiver, Sender, TryRecvError};
use std::thread;
use std::time::Duration;

use tungstenite::{Error, Message};

pub enum Ev {
    Msg(Vec<u8>),
    Closed(String),
}

pub struct Net {
    pub out: Sender<Vec<u8>>,
    pub rx: Receiver<Ev>,
}

fn host_port(url: &str) -> String {
    let rest = url.split("://").last().unwrap_or(url);
    rest.split('/').next().unwrap_or(rest).to_string()
}

pub fn connect(url: &str) -> Net {
    let (out, out_rx) = channel::<Vec<u8>>();
    let (tx, rx) = channel::<Ev>();
    let url = if url.contains("://") { url.to_string() } else { format!("ws://{url}") };
    thread::spawn(move || {
        let res = run(&url, out_rx, &tx);
        let _ = tx.send(Ev::Closed(match res {
            Ok(()) => "closed".into(),
            Err(e) => e,
        }));
    });
    Net { out, rx }
}

fn would_block(e: &Error) -> bool {
    matches!(e, Error::Io(e) if matches!(e.kind(), ErrorKind::WouldBlock | ErrorKind::TimedOut))
}

fn run(url: &str, out_rx: Receiver<Vec<u8>>, tx: &Sender<Ev>) -> Result<(), String> {
    let stream = TcpStream::connect(host_port(url)).map_err(|e| e.to_string())?;
    stream.set_nodelay(true).ok();
    let (mut ws, _) = tungstenite::client(url, stream).map_err(|e| e.to_string())?;
    ws.get_ref().set_read_timeout(Some(Duration::from_millis(1))).ok();
    for hello in [[254u8, 6, 0, 0, 0], [255, 1, 0, 0, 0]] {
        ws.send(Message::Binary(hello.to_vec().into())).map_err(|e| e.to_string())?;
    }
    loop {
        loop {
            match out_rx.try_recv() {
                Ok(p) => match ws.write(Message::Binary(p.into())) {
                    Ok(()) => {}
                    Err(e) if would_block(&e) => {}
                    Err(e) => return Err(e.to_string()),
                },
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => return Ok(()),
            }
        }
        match ws.flush() {
            Err(e) if !would_block(&e) => return Err(e.to_string()),
            _ => {}
        }
        match ws.read() {
            Ok(Message::Binary(b)) => {
                if tx.send(Ev::Msg(b.to_vec())).is_err() {
                    return Ok(());
                }
            }
            Ok(Message::Close(_)) => return Ok(()),
            Ok(_) => {}
            Err(e) if would_block(&e) => {}
            Err(e) => return Err(e.to_string()),
        }
    }
}
