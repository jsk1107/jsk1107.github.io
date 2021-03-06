!function(a) {
    "function" == typeof require && "object" == typeof exports ? module.exports = a() : "function" == typeof define && define.amd ? define([ "velocity" ], a) : a();
}(function() {
    return function(a, b, c, d) {
        function e(a, b) {
            var c = [];
            return a && b ? (g.each([ a, b ], function(a, b) {
                var d = [];
                g.each(b, function(a, b) {
                    for (;b.toString().length < 5; ) b = "0" + b;
                    d.push(b);
                }), c.push(d.join(""));
            }), parseFloat(c[0]) > parseFloat(c[1])) : !1;
        }
        if (!a.Velocity || !a.Velocity.Utilities) return void (b.console && console.log("Velocity UI Pack: Velocity must be loaded first. Aborting."));
        var f = a.Velocity, g = f.Utilities, h = f.version, i = {
            major: 1,
            minor: 1,
            patch: 0
        };
        if (e(i, h)) {
            var j = "Velocity UI Pack: You need to update Velocity (jquery.velocity.js) to a newer version. Visit http://github.com/julianshapiro/velocity.";
            throw alert(j), new Error(j);
        }
        f.RegisterEffect = f.RegisterUI = function(a, b) {
            function c(a, b, c, d) {
                var e = 0, h;
                g.each(a.nodeType ? [ a ] : a, function(a, b) {
                    d && (c += a * d), h = b.parentNode, g.each([ "height", "paddingTop", "paddingBottom", "marginTop", "marginBottom" ], function(a, c) {
                        e += parseFloat(f.CSS.getPropertyValue(b, c));
                    });
                }), f.animate(h, {
                    height: ("In" === b ? "+" : "-") + "=" + e
                }, {
                    queue: !1,
                    easing: "ease-in-out",
                    duration: c * ("In" === b ? .6 : 1)
                });
            }
            return f.Redirects[a] = function(e, h, i, j, k, l) {
                function m() {
                    h.display !== d && "none" !== h.display || !/Out$/.test(a) || g.each(k.nodeType ? [ k ] : k, function(a, b) {
                        f.CSS.setPropertyValue(b, "display", "none");
                    }), h.complete && h.complete.call(k, k), l && l.resolver(k || e);
                }
                var n = i === j - 1;
                b.defaultDuration = "function" == typeof b.defaultDuration ? b.defaultDuration.call(k, k) : parseFloat(b.defaultDuration);
                for (var o = 0; o < b.calls.length; o++) {
                    var p = b.calls[o], q = p[0], r = h.duration || b.defaultDuration || 1e3, s = p[1], t = p[2] || {}, u = {};
                    if (u.duration = r * (s || 1), u.queue = h.queue || "", u.easing = t.easing || "ease", 
                    u.delay = parseFloat(t.delay) || 0, u._cacheValues = t._cacheValues || !0, 0 === o) {
                        if (u.delay += parseFloat(h.delay) || 0, 0 === i && (u.begin = function() {
                            h.begin && h.begin.call(k, k);
                            var b = a.match(/(In|Out)$/);
                            b && "In" === b[0] && q.opacity !== d && g.each(k.nodeType ? [ k ] : k, function(a, b) {
                                f.CSS.setPropertyValue(b, "opacity", 0);
                            }), h.animateParentHeight && b && c(k, b[0], r + u.delay, h.stagger);
                        }), null !== h.display) if (h.display !== d && "none" !== h.display) u.display = h.display; else if (/In$/.test(a)) {
                            var v = f.CSS.Values.getDisplayType(e);
                            u.display = "inline" === v ? "inline-block" : v;
                        }
                        h.visibility && "hidden" !== h.visibility && (u.visibility = h.visibility);
                    }
                    o === b.calls.length - 1 && (u.complete = function() {
                        if (b.reset) {
                            for (var a in b.reset) {
                                var c = b.reset[a];
                                f.CSS.Hooks.registered[a] !== d || "string" != typeof c && "number" != typeof c || (b.reset[a] = [ b.reset[a], b.reset[a] ]);
                            }
                            var g = {
                                duration: 0,
                                queue: !1
                            };
                            n && (g.complete = m), f.animate(e, b.reset, g);
                        } else n && m();
                    }, "hidden" === h.visibility && (u.visibility = h.visibility)), f.animate(e, q, u);
                }
            }, f;
        }, f.RegisterEffect.packagedEffects = {
            "callout.bounce": {
                defaultDuration: 550,
                calls: [ [ {
                    translateY: -30
                }, .25 ], [ {
                    translateY: 0
                }, .125 ], [ {
                    translateY: -15
                }, .125 ], [ {
                    translateY: 0
                }, .25 ] ]
            },
            "callout.shake": {
                defaultDuration: 800,
                calls: [ [ {
                    translateX: -11
                }, .125 ], [ {
                    translateX: 11
                }, .125 ], [ {
                    translateX: -11
                }, .125 ], [ {
                    translateX: 11
                }, .125 ], [ {
                    translateX: -11
                }, .125 ], [ {
                    translateX: 11
                }, .125 ], [ {
                    translateX: -11
                }, .125 ], [ {
                    translateX: 0
                }, .125 ] ]
            },
            "callout.flash": {
                defaultDuration: 1100,
                calls: [ [ {
                    opacity: [ 0, "easeInOutQuad", 1 ]
                }, .25 ], [ {
                    opacity: [ 1, "easeInOutQuad" ]
                }, .25 ], [ {
                    opacity: [ 0, "easeInOutQuad" ]
                }, .25 ], [ {
                    opacity: [ 1, "easeInOutQuad" ]
                }, .25 ] ]
            },
            "callout.pulse": {
                defaultDuration: 825,
                calls: [ [ {
                    scaleX: 1.1,
                    scaleY: 1.1
                }, .5, {
                    easing: "easeInExpo"
                } ], [ {
                    scaleX: 1,
                    scaleY: 1
                }, .5 ] ]
            },
            "callout.swing": {
                defaultDuration: 950,
                calls: [ [ {
                    rotateZ: 15
                }, .2 ], [ {
                    rotateZ: -10
                }, .2 ], [ {
                    rotateZ: 5
                }, .2 ], [ {
                    rotateZ: -5
                }, .2 ], [ {
                    rotateZ: 0
                }, .2 ] ]
            },
            "callout.tada": {
                defaultDuration: 1e3,
                calls: [ [ {
                    scaleX: .9,
                    scaleY: .9,
                    rotateZ: -3
                }, .1 ], [ {
                    scaleX: 1.1,
                    scaleY: 1.1,
                    rotateZ: 3
                }, .1 ], [ {
                    scaleX: 1.1,
                    scaleY: 1.1,
                    rotateZ: -3
                }, .1 ], [ "reverse", .125 ], [ "reverse", .125 ], [ "reverse", .125 ], [ "reverse", .125 ], [ "reverse", .125 ], [ {
                    scaleX: 1,
                    scaleY: 1,
                    rotateZ: 0
                }, .2 ] ]
            },
            "transition.fadeIn": {
                defaultDuration: 500,
                calls: [ [ {
                    opacity: [ 1, 0 ]
                } ] ]
            },
            "transition.fadeOut": {
                defaultDuration: 500,
                calls: [ [ {
                    opacity: [ 0, 1 ]
                } ] ]
            },
            "transition.flipXIn": {
                defaultDuration: 700,
                calls: [ [ {
                    opacity: [ 1, 0 ],
                    transformPerspective: [ 800, 800 ],
                    rotateY: [ 0, -55 ]
                } ] ],
                reset: {
                    transformPerspective: 0
                }
            },
            "transition.flipXOut": {
                defaultDuration: 700,
                calls: [ [ {
                    opacity: [ 0, 1 ],
                    transformPerspective: [ 800, 800 ],
                    rotateY: 55
                } ] ],
                reset: {
                    transformPerspective: 0,
                    rotateY: 0
                }
            },
            "transition.flipYIn": {
                defaultDuration: 800,
                calls: [ [ {
                    opacity: [ 1, 0 ],
                    transformPerspective: [ 800, 800 ],
                    rotateX: [ 0, -45 ]
                } ] ],
                reset: {
                    transformPerspective: 0
                }
            },
            "transition.flipYOut": {
                defaultDuration: 800,
                calls: [ [ {
                    opacity: [ 0, 1 ],
                    transformPerspective: [ 800, 800 ],
                    rotateX: 25
                } ] ],
                reset: {
                    transformPerspective: 0,
                    rotateX: 0
                }
            },
            "transition.flipBounceXIn": {
                defaultDuration: 900,
                calls: [ [ {
                    opacity: [ .725, 0 ],
                    transformPerspective: [ 400, 400 ],
                    rotateY: [ -10, 90 ]
                }, .5 ], [ {
                    opacity: .8,
                    rotateY: 10
                }, .25 ], [ {
                    opacity: 1,
                    rotateY: 0
                }, .25 ] ],
                reset: {
                    transformPerspective: 0
                }
            },
            "transition.flipBounceXOut": {
                defaultDuration: 800,
                calls: [ [ {
                    opacity: [ .9, 1 ],
                    transformPerspective: [ 400, 400 ],
                    rotateY: -10
                }, .5 ], [ {
                    opacity: 0,
                    rotateY: 90
                }, .5 ] ],
                reset: {
                    transformPerspective: 0,
                    rotateY: 0
                }
            },
            "transition.flipBounceYIn": {
                defaultDuration: 850,
                calls: [ [ {
                    opacity: [ .725, 0 ],
                    transformPerspective: [ 400, 400 ],
                    rotateX: [ -10, 90 ]
                }, .5 ], [ {
                    opacity: .8,
                    rotateX: 10
                }, .25 ], [ {
                    opacity: 1,
                    rotateX: 0
                }, .25 ] ],
                reset: {
                    transformPerspective: 0
                }
            },
            "transition.flipBounceYOut": {
                defaultDuration: 800,
                calls: [ [ {
                    opacity: [ .9, 1 ],
                    transformPerspective: [ 400, 400 ],
                    rotateX: -15
                }, .5 ], [ {
                    opacity: 0,
                    rotateX: 90
                }, .5 ] ],
                reset: {
                    transformPerspective: 0,
                    rotateX: 0
                }
            },
            "transition.swoopIn": {
                defaultDuration: 850,
                calls: [ [ {
                    opacity: [ 1, 0 ],
                    transformOriginX: [ "100%", "50%" ],
                    transformOriginY: [ "100%", "100%" ],
                    scaleX: [ 1, 0 ],
                    scaleY: [ 1, 0 ],
                    translateX: [ 0, -700 ],
                    translateZ: 0
                } ] ],
                reset: {
                    transformOriginX: "50%",
                    transformOriginY: "50%"
                }
            },
            "transition.swoopOut": {
                defaultDuration: 850,
                calls: [ [ {
                    opacity: [ 0, 1 ],
                    transformOriginX: [ "50%", "100%" ],
                    transformOriginY: [ "100%", "100%" ],
                    scaleX: 0,
                    scaleY: 0,
                    translateX: -700,
                    translateZ: 0
                } ] ],
                reset: {
                    transformOriginX: "50%",
                    transformOriginY: "50%",
                    scaleX: 1,
                    scaleY: 1,
                    translateX: 0
                }
            },
            "transition.whirlIn": {
                defaultDuration: 850,
                calls: [ [ {
                    opacity: [ 1, 0 ],
                    transformOriginX: [ "50%", "50%" ],
                    transformOriginY: [ "50%", "50%" ],
                    scaleX: [ 1, 0 ],
                    scaleY: [ 1, 0 ],
                    rotateY: [ 0, 160 ]
                }, 1, {
                    easing: "easeInOutSine"
                } ] ]
            },
            "transition.whirlOut": {
                defaultDuration: 750,
                calls: [ [ {
                    opacity: [ 0, "easeInOutQuint", 1 ],
                    transformOriginX: [ "50%", "50%" ],
                    transformOriginY: [ "50%", "50%" ],
                    scaleX: 0,
                    scaleY: 0,
                    rotateY: 160
                }, 1, {
                    easing: "swing"
                } ] ],
                reset: {
                    scaleX: 1,
                    scaleY: 1,
                    rotateY: 0
                }
            },
            "transition.shrinkIn": {
                defaultDuration: 750,
                calls: [ [ {
                    opacity: [ 1, 0 ],
                    transformOriginX: [ "50%", "50%" ],
                    transformOriginY: [ "50%", "50%" ],
                    scaleX: [ 1, 1.5 ],
                    scaleY: [ 1, 1.5 ],
                    translateZ: 0
                } ] ]
            },
            "transition.shrinkOut": {
                defaultDuration: 600,
                calls: [ [ {
                    opacity: [ 0, 1 ],
                    transformOriginX: [ "50%", "50%" ],
                    transformOriginY: [ "50%", "50%" ],
                    scaleX: 1.3,
                    scaleY: 1.3,
                    translateZ: 0
                } ] ],
                reset: {
                    scaleX: 1,
                    scaleY: 1
                }
            },
            "transition.expandIn": {
                defaultDuration: 700,
                calls: [ [ {
                    opacity: [ 1, 0 ],
                    transformOriginX: [ "50%", "50%" ],
                    transformOriginY: [ "50%", "50%" ],
                    scaleX: [ 1, .625 ],
                    scaleY: [ 1, .625 ],
                    translateZ: 0
                } ] ]
            },
            "transition.expandOut": {
                defaultDuration: 700,
                calls: [ [ {
                    opacity: [ 0, 1 ],
                    transformOriginX: [ "50%", "50%" ],
                    transformOriginY: [ "50%", "50%" ],
                    scaleX: .5,
                    scaleY: .5,
                    translateZ: 0
                } ] ],
                reset: {
                    scaleX: 1,
                    scaleY: 1
                }
            },
            "transition.bounceIn": {
                defaultDuration: 800,
                calls: [ [ {
                    opacity: [ 1, 0 ],
                    scaleX: [ 1.05, .3 ],
                    scaleY: [ 1.05, .3 ]
                }, .4 ], [ {
                    scaleX: .9,
                    scaleY: .9,
                    translateZ: 0
                }, .2 ], [ {
                    scaleX: 1,
                    scaleY: 1
                }, .5 ] ]
            },
            "transition.bounceOut": {
                defaultDuration: 800,
                calls: [ [ {
                    scaleX: .95,
                    scaleY: .95
                }, .35 ], [ {
                    scaleX: 1.1,
                    scaleY: 1.1,
                    translateZ: 0
                }, .35 ], [ {
                    opacity: [ 0, 1 ],
                    scaleX: .3,
                    scaleY: .3
                }, .3 ] ],
                reset: {
                    scaleX: 1,
                    scaleY: 1
                }
            },
            "transition.bounceUpIn": {
                defaultDuration: 800,
                calls: [ [ {
                    opacity: [ 1, 0 ],
                    translateY: [ -30, 1e3 ]
                }, .6, {
                    easing: "easeOutCirc"
                } ], [ {
                    translateY: 10
                }, .2 ], [ {
                    translateY: 0
                }, .2 ] ]
            },
            "transition.bounceUpOut": {
                defaultDuration: 1e3,
                calls: [ [ {
                    translateY: 20
                }, .2 ], [ {
                    opacity: [ 0, "easeInCirc", 1 ],
                    translateY: -1e3
                }, .8 ] ],
                reset: {
                    translateY: 0
                }
            },
            "transition.bounceDownIn": {
                defaultDuration: 800,
                calls: [ [ {
                    opacity: [ 1, 0 ],
                    translateY: [ 30, -1e3 ]
                }, .6, {
                    easing: "easeOutCirc"
                } ], [ {
                    translateY: -10
                }, .2 ], [ {
                    translateY: 0
                }, .2 ] ]
            },
            "transition.bounceDownOut": {
                defaultDuration: 1e3,
                calls: [ [ {
                    translateY: -20
                }, .2 ], [ {
                    opacity: [ 0, "easeInCirc", 1 ],
                    translateY: 1e3
                }, .8 ] ],
                reset: {
                    translateY: 0
                }
            },
            "transition.bounceLeftIn": {
                defaultDuration: 750,
                calls: [ [ {
                    opacity: [ 1, 0 ],
                    translateX: [ 30, -1250 ]
                }, .6, {
                    easing: "easeOutCirc"
                } ], [ {
                    translateX: -10
                }, .2 ], [ {
                    translateX: 0
                }, .2 ] ]
            },
            "transition.bounceLeftOut": {
                defaultDuration: 750,
                calls: [ [ {
                    translateX: 30
                }, .2 ], [ {
                    opacity: [ 0, "easeInCirc", 1 ],
                    translateX: -1250
                }, .8 ] ],
                reset: {
                    translateX: 0
                }
            },
            "transition.bounceRightIn": {
                defaultDuration: 750,
                calls: [ [ {
                    opacity: [ 1, 0 ],
                    translateX: [ -30, 1250 ]
                }, .6, {
                    easing: "easeOutCirc"
                } ], [ {
                    translateX: 10
                }, .2 ], [ {
                    translateX: 0
                }, .2 ] ]
            },
            "transition.bounceRightOut": {
                defaultDuration: 750,
                calls: [ [ {
                    translateX: -30
                }, .2 ], [ {
                    opacity: [ 0, "easeInCirc", 1 ],
                    translateX: 1250
                }, .8 ] ],
                reset: {
                    translateX: 0
                }
            },
            "transition.slideUpIn": {
                defaultDuration: 900,
                calls: [ [ {
                    opacity: [ 1, 0 ],
                    translateY: [ 0, 20 ],
                    translateZ: 0
                } ] ]
            },
            "transition.slideUpOut": {
                defaultDuration: 900,
                calls: [ [ {
                    opacity: [ 0, 1 ],
                    translateY: -20,
                    translateZ: 0
                } ] ],
                reset: {
                    translateY: 0
                }
            },
            "transition.slideDownIn": {
                defaultDuration: 900,
                calls: [ [ {
                    opacity: [ 1, 0 ],
                    translateY: [ 0, -20 ],
                    translateZ: 0
                } ] ]
            },
            "transition.slideDownOut": {
                defaultDuration: 900,
                calls: [ [ {
                    opacity: [ 0, 1 ],
                    translateY: 20,
                    translateZ: 0
                } ] ],
                reset: {
                    translateY: 0
                }
            },
            "transition.slideLeftIn": {
                defaultDuration: 1e3,
                calls: [ [ {
                    opacity: [ 1, 0 ],
                    translateX: [ 0, -20 ],
                    translateZ: 0
                } ] ]
            },
            "transition.slideLeftOut": {
                defaultDuration: 1050,
                calls: [ [ {
                    opacity: [ 0, 1 ],
                    translateX: -20,
                    translateZ: 0
                } ] ],
                reset: {
                    translateX: 0
                }
            },
            "transition.slideRightIn": {
                defaultDuration: 1e3,
                calls: [ [ {
                    opacity: [ 1, 0 ],
                    translateX: [ 0, 20 ],
                    translateZ: 0
                } ] ]
            },
            "transition.slideRightOut": {
                defaultDuration: 1050,
                calls: [ [ {
                    opacity: [ 0, 1 ],
                    translateX: 20,
                    translateZ: 0
                } ] ],
                reset: {
                    translateX: 0
                }
            },
            "transition.slideUpBigIn": {
                defaultDuration: 850,
                calls: [ [ {
                    opacity: [ 1, 0 ],
                    translateY: [ 0, 75 ],
                    translateZ: 0
                } ] ]
            },
            "transition.slideUpBigOut": {
                defaultDuration: 800,
                calls: [ [ {
                    opacity: [ 0, 1 ],
                    translateY: -75,
                    translateZ: 0
                } ] ],
                reset: {
                    translateY: 0
                }
            },
            "transition.slideDownBigIn": {
                defaultDuration: 850,
                calls: [ [ {
                    opacity: [ 1, 0 ],
                    translateY: [ 0, -75 ],
                    translateZ: 0
                } ] ]
            },
            "transition.slideDownBigOut": {
                defaultDuration: 800,
                calls: [ [ {
                    opacity: [ 0, 1 ],
                    translateY: 75,
                    translateZ: 0
                } ] ],
                reset: {
                    translateY: 0
                }
            },
            "transition.slideLeftBigIn": {
                defaultDuration: 800,
                calls: [ [ {
                    opacity: [ 1, 0 ],
                    translateX: [ 0, -75 ],
                    translateZ: 0
                } ] ]
            },
            "transition.slideLeftBigOut": {
                defaultDuration: 750,
                calls: [ [ {
                    opacity: [ 0, 1 ],
                    translateX: -75,
                    translateZ: 0
                } ] ],
                reset: {
                    translateX: 0
                }
            },
            "transition.slideRightBigIn": {
                defaultDuration: 800,
                calls: [ [ {
                    opacity: [ 1, 0 ],
                    translateX: [ 0, 75 ],
                    translateZ: 0
                } ] ]
            },
            "transition.slideRightBigOut": {
                defaultDuration: 750,
                calls: [ [ {
                    opacity: [ 0, 1 ],
                    translateX: 75,
                    translateZ: 0
                } ] ],
                reset: {
                    translateX: 0
                }
            },
            "transition.perspectiveUpIn": {
                defaultDuration: 800,
                calls: [ [ {
                    opacity: [ 1, 0 ],
                    transformPerspective: [ 800, 800 ],
                    transformOriginX: [ 0, 0 ],
                    transformOriginY: [ "100%", "100%" ],
                    rotateX: [ 0, -180 ]
                } ] ],
                reset: {
                    transformPerspective: 0,
                    transformOriginX: "50%",
                    transformOriginY: "50%"
                }
            },
            "transition.perspectiveUpOut": {
                defaultDuration: 850,
                calls: [ [ {
                    opacity: [ 0, 1 ],
                    transformPerspective: [ 800, 800 ],
                    transformOriginX: [ 0, 0 ],
                    transformOriginY: [ "100%", "100%" ],
                    rotateX: -180
                } ] ],
                reset: {
                    transformPerspective: 0,
                    transformOriginX: "50%",
                    transformOriginY: "50%",
                    rotateX: 0
                }
            },
            "transition.perspectiveDownIn": {
                defaultDuration: 800,
                calls: [ [ {
                    opacity: [ 1, 0 ],
                    transformPerspective: [ 800, 800 ],
                    transformOriginX: [ 0, 0 ],
                    transformOriginY: [ 0, 0 ],
                    rotateX: [ 0, 180 ]
                } ] ],
                reset: {
                    transformPerspective: 0,
                    transformOriginX: "50%",
                    transformOriginY: "50%"
                }
            },
            "transition.perspectiveDownOut": {
                defaultDuration: 850,
                calls: [ [ {
                    opacity: [ 0, 1 ],
                    transformPerspective: [ 800, 800 ],
                    transformOriginX: [ 0, 0 ],
                    transformOriginY: [ 0, 0 ],
                    rotateX: 180
                } ] ],
                reset: {
                    transformPerspective: 0,
                    transformOriginX: "50%",
                    transformOriginY: "50%",
                    rotateX: 0
                }
            },
            "transition.perspectiveLeftIn": {
                defaultDuration: 950,
                calls: [ [ {
                    opacity: [ 1, 0 ],
                    transformPerspective: [ 2e3, 2e3 ],
                    transformOriginX: [ 0, 0 ],
                    transformOriginY: [ 0, 0 ],
                    rotateY: [ 0, -180 ]
                } ] ],
                reset: {
                    transformPerspective: 0,
                    transformOriginX: "50%",
                    transformOriginY: "50%"
                }
            },
            "transition.perspectiveLeftOut": {
                defaultDuration: 950,
                calls: [ [ {
                    opacity: [ 0, 1 ],
                    transformPerspective: [ 2e3, 2e3 ],
                    transformOriginX: [ 0, 0 ],
                    transformOriginY: [ 0, 0 ],
                    rotateY: -180
                } ] ],
                reset: {
                    transformPerspective: 0,
                    transformOriginX: "50%",
                    transformOriginY: "50%",
                    rotateY: 0
                }
            },
            "transition.perspectiveRightIn": {
                defaultDuration: 950,
                calls: [ [ {
                    opacity: [ 1, 0 ],
                    transformPerspective: [ 2e3, 2e3 ],
                    transformOriginX: [ "100%", "100%" ],
                    transformOriginY: [ 0, 0 ],
                    rotateY: [ 0, 180 ]
                } ] ],
                reset: {
                    transformPerspective: 0,
                    transformOriginX: "50%",
                    transformOriginY: "50%"
                }
            },
            "transition.perspectiveRightOut": {
                defaultDuration: 950,
                calls: [ [ {
                    opacity: [ 0, 1 ],
                    transformPerspective: [ 2e3, 2e3 ],
                    transformOriginX: [ "100%", "100%" ],
                    transformOriginY: [ 0, 0 ],
                    rotateY: 180
                } ] ],
                reset: {
                    transformPerspective: 0,
                    transformOriginX: "50%",
                    transformOriginY: "50%",
                    rotateY: 0
                }
            }
        };
        for (var k in f.RegisterEffect.packagedEffects) f.RegisterEffect(k, f.RegisterEffect.packagedEffects[k]);
        f.RunSequence = function(a) {
            var b = g.extend(!0, [], a);
            b.length > 1 && (g.each(b.reverse(), function(a, c) {
                var d = b[a + 1];
                if (d) {
                    var e = c.o || c.options, h = d.o || d.options, i = e && e.sequenceQueue === !1 ? "begin" : "complete", j = h && h[i], k = {};
                    k[i] = function() {
                        var a = d.e || d.elements, b = a.nodeType ? [ a ] : a;
                        j && j.call(b, b), f(c);
                    }, d.o ? d.o = g.extend({}, h, k) : d.options = g.extend({}, h, k);
                }
            }), b.reverse()), f(b[0]);
        };
    }(window.jQuery || window.Zepto || window, window, document);
});