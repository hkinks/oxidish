"use strict";
// Arena launcher for vendor/MultiOgar: config from CLI, seeded RNG, and a
// lockstep mode for training. Monkeypatches the server instead of editing it.
//
//   node arena.js --port 9001 --bots 8 --map 7000 --sync 2 --seed 1 [--agents 4] [--max-tps 0] [--realtime]
//
// Lockstep: clients that send 0xC8 are "agents". The server runs --sync ticks,
// sends each agent 0xC9 <uint32 tick>, then waits until every agent replies
// 0xC9 <same tick> (after sending its mouse/split), or ACK_TIMEOUT_MS passes.
// Other clients (e.g. a spectating browser) are unaffected.
// With --agents N the first step waits until N agents have joined.
// Game rules count ticks, not wall time, so stepping faster than real time
// doesn't change the physics.

const path = require("path");

const args = {port: 9001, bots: 8, map: 7000, sync: 2, seed: 0, agents: 0, "max-tps": 0,
              realtime: false, food: 0, verbose: false, follow: "", "log-players": 0,
              "food-mult": 1, minions: 0};
for (let i = 2; i < process.argv.length; i++) {
    const key = process.argv[i].replace(/^--/, "");
    if (!(key in args)) throw new Error(`unknown option --${key}`);
    if (typeof args[key] === "boolean") args[key] = true;
    else if (typeof args[key] === "string") args[key] = process.argv[++i];
    else args[key] = Number(process.argv[++i]);
}

if (args.seed) { // mulberry32: every Math.random() in the server becomes reproducible
    let a = args.seed >>> 0;
    Math.random = () => {
        a = (a + 0x6D2B79F5) >>> 0;
        let t = a;
        t = Math.imul(t ^ (t >>> 15), t | 1);
        t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
        return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
}

const SRC = path.join(__dirname, "../../vendor/MultiOgar/src");
process.chdir(SRC); // the server reads ../src/config.ini and txt files relative to cwd
const Log = require(path.join(SRC, "modules/Logger"));
const GameServer = require(path.join(SRC, "GameServer"));
const PacketHandler = require(path.join(SRC, "PacketHandler"));
const QuadNode = require(path.join(SRC, "modules/QuadNode"));
const PlayerTracker = require(path.join(SRC, "PlayerTracker"));

process.stdout.on("error", () => process.exit(0)); // parent closed the pipe: match is over

const OP_AGENT = 0xC8, OP_STEP = 0xC9, OP_FOLLOW = 0xCA, OP_SPAWN_SIZE = 0xCB, OP_FOOD = 0xCC;
const OP_GROW = 0xCD;   // human cheat key (+ in the web client): mass onto our biggest cell
const GROW_MASS = 500;
const ACK_TIMEOUT_MS = 2000;

Log.start();
const gs = new GameServer();
Log.setVerbosity(args.verbose ? 4 : 2);
Log.setFileVerbosity(0);
Object.assign(gs.config, {
    serverPort: args.port,
    serverBots: args.bots,
    serverIpLimit: 0,       // all agents connect from localhost
    scrambleLevel: 0,       // plain world coordinates
    serverTimeout: 0,       // wall-clock timeout is meaningless when time is stepped
    serverChat: 0,
    minionChecking: 0,      // flags non-browser clients; all our agents are headless
    borderWidth: args.map,
    borderHeight: args.map,
});
// Keep food density of the default 14142 map unless --food is given.
const area = (args.map / 14142) ** 2;
const baseFood = args.food || Math.round(1500 * area);
gs.config.foodMinAmount = Math.round(baseFood * args["food-mult"]);
gs.config.foodMaxAmount = Math.round(gs.config.foodMinAmount * 1.3);
gs.config.virusMinAmount = Math.max(4, Math.round(50 * area));
gs.config.virusMaxAmount = Math.max(gs.config.virusMinAmount + 2, Math.round(100 * area));
gs.setBorder(args.map, args.map);
gs.quadTree = new QuadNode(gs.border, 64, 32);

// Virus attribution, reported to agents in the step packet: who fed a virus until it
// fired ("shots"), and whose fired virus another player then exploded on ("pops").
const Virus = require(path.join(SRC, "entity/Virus"));
const vstats = p => p.vstats || (p.vstats = {shots: 0, pops: 0, popMass: 0});
const origAddNode = gs.addNode.bind(gs);
gs.addNode = function (node) {
    if (this._shooter && (node.cellType === 2 || node.cellType === 3)) node.shooter = this._shooter;
    return origAddNode(node);
};
const origEjectMass = gs.ejectMass.bind(gs);
gs.ejectMass = function (client) {
    this._shooter = client;
    try { return origEjectMass(client); } finally { this._shooter = null; }
};
const origShootVirus = gs.shootVirus.bind(gs);
gs.shootVirus = function (parent, angle) {
    const by = parent.lastFeeder || null;
    if (by) vstats(by).shots++;
    this._shooter = by;
    try { return origShootVirus(parent, angle); } finally { this._shooter = null; }
};
const origVirusEat = Virus.prototype.onEat;
Virus.prototype.onEat = function (cell) {
    if (cell.shooter) this.lastFeeder = cell.shooter;
    return origVirusEat.call(this, cell);
};
const origVirusEaten = Virus.prototype.onEaten;
Virus.prototype.onEaten = function (cell) {
    const owner = cell.owner, cfg = this.gameServer.config;
    if (this.shooter && owner && owner !== this.shooter
        && (cfg.virusMaxCells || cfg.playerMaxCells) - owner.cells.length > 0) {
        const s = vstats(this.shooter);
        s.pops++;
        s.popMass += cell._mass;
    }
    return origVirusEaten.call(this, cell);
};

const origHandle = PacketHandler.prototype.handleMessage;
PacketHandler.prototype.handleMessage = function (message) {
    const ws = this.socket;
    if (message[0] === OP_AGENT && message.length === 1) {
        ws.isAgent = true;
        ws.ackTick = -1;
        return;
    }
    if (message[0] === OP_FOLLOW) { // spectator picks whom to follow: name prefix, "" = auto
        ws.playerTracker.followName = message.slice(1).toString("utf8");
        return;
    }
    if (message[0] === OP_FOOD && message.length === 3) { // live food density: u16 = multiplier * 100
        gs.config.foodMinAmount = Math.round(baseFood * message.readUInt16LE(1) / 100);
        gs.config.foodMaxAmount = Math.round(gs.config.foodMinAmount * 1.3);
        process.stdout.write(`[arena] food target now ${gs.config.foodMinAmount}\n`);
        return;
    }
    if (message[0] === OP_GROW && message.length === 1 && !ws.isAgent) {
        const pt = ws.playerTracker;
        if (!pt.cells.length) return;
        const cell = pt.cells.reduce((a, b) => (b._size > a._size ? b : a));
        cell.setSize(Math.sqrt(cell._size * cell._size + GROW_MASS * 100));  // mass = size^2 / 100
        // Tag the name for the rest of this life so showcase stats keep cheated mass apart
        // from the human benchmark (a respawn sends the plain name again).
        if (!(pt._name || "").endsWith(" (+)")) pt.setName((pt._name || "") + " (+)");
        return;
    }
    if (message[0] === 0 && !ws.isAgent && args.minions > 0 && !ws.minionsGiven) {
        // Humans (web client) get minions on their first join; agents never do, since they
        // announce themselves with OP_AGENT before joining. Keys: E split, R eject, Q follow,
        // T freeze, P collect food. Minions are named "minion" to keep them out of the
        // human's showcase stats.
        origHandle.call(this, message);
        ws.minionsGiven = true;
        for (let i = 0; i < args.minions; i++) gs.bots.addMinion(ws.playerTracker, "minion");
        ws.playerTracker.minion.control = true;
        return;
    }
    if (message[0] === OP_SPAWN_SIZE && message.length === 3) { // next spawn's cell size (0 = default)
        ws.playerTracker.spawnMass = message.readUInt16LE(1);   // MultiOgar treats it as a size
        return;
    }
    if (message[0] === OP_STEP && message.length === 5) {
        ws.ackTick = message.readUInt32LE(1);
        ws.lastAliveTime = Date.now();
        maybeAdvance();
        return;
    }
    return origHandle.call(this, message);
};

// Spectators follow the largest living player whose name starts with their chosen
// prefix (OP_FOLLOW from the watch panel, else --follow); otherwise the largest player.
{
    const origSpec = PlayerTracker.prototype.getSpecTarget;
    PlayerTracker.prototype.getSpecTarget = function () {
        const prefix = this.followName !== undefined ? this.followName : args.follow;
        if (!prefix) return origSpec.call(this);
        let best = null, bestMass = -1;
        for (const ws of gs.clients) {
            const pt = ws.playerTracker;
            if (!pt || pt.isRemoved || !pt.cells.length || !(pt._name || "").startsWith(prefix)) continue;
            const mass = pt.cells.reduce((m, c) => m + c._mass, 0);
            if (mass > bestMass) { best = pt; bestMass = mass; }
        }
        return best || origSpec.call(this);
    };
}

const agents = () => gs.clients.filter(ws => ws.isAgent && ws.isConnected !== false);
let pending = false, ackTimer = null, lastStepAt = 0, started = false;
let steps = 0, timeouts = 0, statAt = Date.now(), statTicks = 0;

function maybeAdvance() {
    if (!pending) return;
    if (agents().every(ws => ws.ackTick >= gs.tickCount)) advance();
}

function advance() {
    pending = false;
    clearTimeout(ackTimer);
    setImmediate(step);
}

function step() {
    const list = agents();
    if (!started) {
        if (list.length < args.agents) return setTimeout(step, 20);
        started = true;
    }
    if (args["max-tps"] > 0) {
        const wait = lastStepAt + args.sync * 1000 / args["max-tps"] - Date.now();
        if (wait > 1) return setTimeout(step, wait);
    }
    lastStepAt = Date.now();
    for (let k = 0; k < args.sync; k++) gs.mainLoop();
    steps++;
    statTicks += args.sync;
    if (!list.length) return setTimeout(step, args.sync * gs.config.serverTimeStep);
    for (const ws of list) { // 0xC9 tick, then this agent's cumulative virus shots/pops/popped mass
        const v = vstats(ws.playerTracker), packet = Buffer.alloc(17);
        packet[0] = OP_STEP;
        packet.writeUInt32LE(gs.tickCount, 1);
        packet.writeUInt32LE(v.shots, 5);
        packet.writeUInt32LE(v.pops, 9);
        packet.writeFloatLE(v.popMass, 13);
        ws.send(packet, {binary: true});
    }
    pending = true;
    ackTimer = setTimeout(() => { timeouts++; advance(); }, ACK_TIMEOUT_MS);
}

if (!args.realtime) gs.timerLoop = () => {}; // must precede start(), which binds timerLoop
gs.start();
if (!args.realtime) {
    setTimeout(step, 50);
    setInterval(() => {
        const now = Date.now();
        const tps = statTicks / ((now - statAt) / 1000);
        process.stdout.write(`[arena] tick=${gs.tickCount} tps=${tps.toFixed(0)} ` +
            `(${(tps / 25).toFixed(1)}x realtime) agents=${agents().length} timeouts=${timeouts}\n`);
        statAt = now;
        statTicks = 0;
    }, 5000);
}
if (args["log-players"] > 0) {
    // Every N seconds of game time: every live player's name, mass and cell count, as JSON.
    const every = Math.round(args["log-players"] * 25);
    const origMain = gs.mainLoop.bind(gs);
    gs.mainLoop = function () {
        origMain();
        if (gs.tickCount % every !== 0) return;
        const players = [];
        for (const ws of gs.clients) {
            const pt = ws.playerTracker;
            if (!pt || pt.isRemoved || !pt.cells.length) continue;
            players.push({n: pt._name || "", m: Math.round(pt.cells.reduce((s, c) => s + c._mass, 0)),
                          c: pt.cells.length});
        }
        process.stdout.write(`[players] ${JSON.stringify({t: gs.tickCount, p: players})}\n`);
    };
}

process.stdout.write(`[arena] port=${args.port} map=${args.map} bots=${args.bots} sync=${args.sync} ` +
                     `seed=${args.seed} mode=${args.realtime ? "realtime" : "lockstep"}\n`);
