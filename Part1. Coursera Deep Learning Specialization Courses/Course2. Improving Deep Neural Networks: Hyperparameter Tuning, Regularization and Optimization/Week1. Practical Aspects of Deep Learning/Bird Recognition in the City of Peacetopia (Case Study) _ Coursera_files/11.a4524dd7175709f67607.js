(window.webpackJsonp=window.webpackJsonp||[]).push([[11],{"+M5F":function(module,e,t){"use strict";t.d(e,"a",function(){return f});var r=t("S+eF"),n=t.n(r),a=t("JAlB"),o=t("wDa7"),s=t("uJqh"),i=t("ML/G"),u=t("GnyC"),c=function receiveDeadlines(e,t){var r=t.deadlines,n=r.isEnabled,a=r.moduleDeadlines;n?e.dispatch("LOAD_COURSE_DEADLINES",{moduleDeadlines:a}):e.dispatch("DISABLE_DEADLINES")},d=function enableDeadlines(e){var t=e.getStore("CourseStore").getCourseId();return o.a.sendStartTime(!0,t).fail(function(e){throw e}).then(s.a).then(function(t){var r=t.elements,n=r[0].start;return i.a.pushV2(["open_course_home.welcome.emit.course_deadline_set",{first_week_due_time:n}]),e.executeAction(c,{deadlines:r[0]})})},l=function setDeadlinesIfEligible(e){var t=e.getStore("CourseStore"),r=e.getStore("CourseScheduleStore"),o=e.getStore("ProgressStore"),s=Object(u.a)(t,r,o),i=t.getCourseId(),c=e.getStore("CourseMembershipStore").isEnrolled(),l=e.getStore("SessionStore");return Object(a.a)(i)||l.isSessionsCourse()||1!==s||!c?n()():e.executeAction(d,{})},f=function loadCourseDeadlines(e,t){var r=t.userId,i=e.getStore("CourseStore").getCourseId(),u=e.getStore("CourseMembershipStore").isEnrolled(),d=e.getStore("SessionStore"),f=e.getStore("CourseStore").isReal();if(!u||!r||Object(a.a)(i))return n()();if(d.isSessionsEnabled()){if(d.isEnrolled()){var S=d.getSession(),p={moduleDeadlines:S.moduleDeadlines};p.itemDeadlines=S.itemDeadlines,e.dispatch("LOAD_COURSE_DEADLINES",p)}return n()()}return o.a.getStartTime(i).then(s.a).then(function(t){var r,n=t.elements[0];return n?e.executeAction(c,{deadlines:n}):e.executeAction(l,{})})},S=function disableDeadlines(e){var t=e.getStore("CourseStore").getCourseId();return o.a.disableDeadlines(t).then(function(){return e.dispatch("DISABLE_DEADLINES")}).fail(function(e){throw e})},p=function resetDeadlines(e,t){var r=t.userId,n=e.getStore("CourseStore").getCourseId();return o.a.resetDeadlines(n).then(function(){return e.executeAction(f,{userId:r})}).fail(function(e){throw e})}},"5Ujo":function(module,e,t){"use strict";t.d(e,"a",function(){return o});var r=t("S+eF"),n=t.n(r),a=t("Vu1r"),o=function loadCoursePresentGrade(e,t){var r=t.userId,o=t.courseId;if(e.getStore("CoursePresentGradeStore").hasLoaded())return n()();return r?n()(a.a.getPresentGrade({userId:r,courseId:o})).then(function(t){var r=t.elements[0];e.dispatch("LOAD_COURSE_PRESENT_GRADE",{presentGrade:r})}).fail(function(t){e.dispatch("LOAD_COURSE_PRESENT_GRADE_FAIL",{})}):(e.dispatch("LOAD_COURSE_PRESENT_GRADE_FAIL",{}),n()())}},BYIE:function(module,e,t){"use strict";var r=t("sArp"),n=t("uJqh");e.a=function(e){if(!e)throw new Error("`courseId` is required to get course schedule.");return Object(r.a)(e).then(n.a).then(function(e){var t;return e.elements[0].defaultSchedule.periods})}},Bgx3:function(module,e,t){"use strict";var r=t("S+eF"),n=t.n(r),a=t("F6uN"),o=t("IG+7");e.a=function(e,t){if(e.getStore(o.a).haveCourseIdentifiersLoaded())return n()();if(!t)throw new Error("Missing courseSlug");return Object(a.a)(t).then(function(r){var n=r.courseId,a=r.courseCertificates,o=r.courseTypeMetadata;if(!n)throw new Error("Missing courseId");return e.dispatch("SET_COURSE_IDENTIFIERS",{courseId:n,courseSlug:t,courseCertificates:a,courseTypeMetadata:o}),{courseId:n,courseSlug:t,courseCertificates:a,courseTypeMetadata:o}}).catch(function(r){console.error("Error getting courseId and courseCertificates from courseSlug: ".concat(t,": "),r,r.stack);var a="",o=[],s={};return e.dispatch("SET_COURSE_IDENTIFIERS",{courseId:"",courseSlug:t,courseCertificates:o,courseTypeMetadata:s}),n()({courseId:"",courseSlug:t,courseCertificates:o})})}},D87r:function(module,e,t){"use strict";t.d(e,"a",function(){return c});var r=t("S+eF"),n=t.n(r),a=t("fw5G"),o=t.n(a),s=t("TSOT"),i=t("sQ/U"),u=Object(s.a)("/api/onDemandHomeProgress.v1",{type:"rest"}),c=function getHomeProgress(e){var t="".concat(i.a.get().id,"~").concat(e),r=new o.a(t).addQueryParam("fields","modulesCompleted,modulesPassed");return n()(u.get(r.toString()))}},DAK3:function(module,e,t){"use strict";t.d(e,"a",function(){return u});var r=t("lvha"),n=t("QU0K"),a=t("Doqk"),o=t("8kE/"),s=function membershipsData(e){return Object(r.b)(a.a.build(n.a.prototype.resourceName,e))},i=s,u=Object(o.a)(s)},F6uN:function(module,e,t){"use strict";var r=t("S+eF"),n=t.n(r),a=t("fw5G"),o=t.n(a),s=t("DnuM"),i=t("15pW");e.a=function(e){var t=Object(s.a)("/api/courses.v1",{type:"rest"}),r=(new o.a).addQueryParam("q","slug").addQueryParam("slug",e).addQueryParam("fields","certificates,courseTypeMetadata.v1(courseTypeMetadata)").addQueryParam("includes","courseTypeMetadata").addQueryParam("showHidden",!0);return n()(t.get(r.toString())).then(function(t){var r;if("notFound"===t.errorCode)return null;var n=t.elements[0],a=n.id,o=n.certificates,s=null===(r=t.linked["courseTypeMetadata.v1"][0])||void 0===r?void 0:r.courseTypeMetadata;return i.d.courseId=a,i.d.courseSlug=e,{courseId:a,courseCertificates:o,courseTypeMetadata:s}})}},FUAI:function(module,e,t){"use strict";t.d(e,"a",function(){return s});var r=t("S+eF"),n=t.n(r),a=t("pkWi"),o=t("5ijc"),s=function loadVerificationDisplay(e,t){var r=t.authenticated,s=t.userId,i=t.courseId,u=t.s12nId,c=t.isCourseVerificationEnabled;if(e.getStore(o.a).hasLoaded())return n()();return r?Object(a.a)(s,i,c,u).then(function(t){e.dispatch("LOAD_VERIFICATION_DISPLAY",t)}):(e.dispatch("LOAD_VERIFICATION_DISPLAY",null),n()())}},JAlB:function(module,e,t){"use strict";var r=t("KMW/");e.a=function(e){return-1!==r.a.get("featureBlacklist","defaultDeadlines").indexOf(e)}},JSqB:function(module,e,t){"use strict";t.d(e,"a",function(){return s}),t.d(e,"b",function(){return i}),t.d(e,"c",function(){return u});var r=t("S+eF"),n=t.n(r),a=t("ROEb"),o=t("tdcm"),s=function loadHonorsUserPreferences(e,t){var r=t.authenticated;if(e.getStore("HonorsUserPreferencesStore").hasLoaded())return n()();return r?o.a.get(o.a.keyEnum.HONORS).then(function(t){e.dispatch("LOAD_HONORS_USER_PREFERENCES",t)}).fail(function(t){e.dispatch("LOAD_HONORS_USER_PREFERENCES",{})}):(e.dispatch("LOAD_HONORS_USER_PREFERENCES",{}),n()())},i=function setHonorsUserPreferences(e,t){var r=t.authenticated,a=t.updatedHonorsUserPreferences;return r?o.a.set(o.a.keyEnum.HONORS,a).then(function(){e.dispatch("LOAD_HONORS_USER_PREFERENCES",a)}):(e.dispatch("LOAD_HONORS_USER_PREFERENCES",a),n()())},u=function setLessonSkipped(e,t){var r=t.lessonId,n=t.skipped;e.dispatch("SET_LESSON_SKIPPED",{lessonId:r,skipped:n})}},JdaY:function(module,e,t){"use strict";t.r(e);var r=t("BJ98"),n=t.n(r),a=t("q1tI"),o=t.n(a),s=t("sQ/U"),i=t("EdUP"),u=t("kwmr"),c=t("+LJP"),d=t("dAof"),l=t("Bgx3"),f=t("E4RX"),S=t("b+2U"),p=t("iTPM"),g=t("Re7p"),h=t("Shko"),v=t("+M5F"),O=t("Nher"),m=t("FUAI"),I=t("dgIx"),E=t("5Ujo"),D=t("JSqB"),b=t("fghW"),L=t("IG+7"),A=t("xPfO"),P=t("5ijc"),w=t("8c4I"),C=t("c2GL"),y=t("sjlm"),j=t("tPFS"),R=t("knci"),F=t("TOZ3"),U=t("8WNh"),k=function DataFetcherBody(e){var t=e.children;if(!t)return null;return o.a.cloneElement(t,{})},M=n()(Object(c.a)(function(e){return{courseSlug:e.params.courseSlug}}),Object(u.a)([A.a,L.a,C.a,b.a,P.a,y.a,w.a,j.a],function(e,t,r,n,a,o,s,i){return{s12n:n.getS12n(),course:t.getMetadata(),courseId:t.getCourseId(),isEnrolled:r.isEnrolled(),sessionId:e.getSessionId(),isEnrolledInSession:e.isEnrolled(),s12nStoreHasLoaded:n.hasLoaded(),courseStoreHasLoaded:t.hasLoaded(),sessionStoreHasLoaded:e.hasLoaded(),verificationStoreHasLoaded:a.hasLoaded(),courseMembershipStoreHasLoaded:r.hasLoaded(),computedModelStoreHasLoaded:s.hasLoaded(),courseIdentifiersHaveLoaded:t.haveCourseIdentifiersLoaded(),courseViewGradeStoreHasLoaded:o.hasLoaded(),progressStoreHasLoaded:i.hasLoaded()}}),Object(d.a)(function(e,t){var r=t.courseSlug;e.executeAction(l.a,r)}),Object(i.a)(function(e){var t;return e.courseIdentifiersHaveLoaded}),Object(i.a)(function(e){var t;return!!e.courseId},o.a.createElement(R.a,null)),Object(d.a)(function(e,t){var r=t.courseId;e.executeAction(h.a,r)}),Object(i.a)(function(e){var t;return e.courseMembershipStoreHasLoaded}),Object(i.a)(function(e){var t=e.isEnrolled;return s.a.isSuperuser()||t},o.a.createElement(R.a,null)),Object(d.a)(function(e,t){var r=t.courseSlug,n=t.courseId;e.executeAction(p.a,{courseSlug:r,courseId:n})}),Object(i.a)(function(e){var t;return e.computedModelStoreHasLoaded}),Object(d.a)(function(e,t){var r=t.courseId,n=t.courseSlug,a=s.a.get().id,o=s.a.isAuthenticatedUser();e.executeAction(S.a,{courseSlug:n}),e.executeAction(O.a,{courseId:r}),e.executeAction(I.a,{courseId:r,userId:a}),e.executeAction(D.a,{authenticated:o}),e.executeAction(g.a,{courseId:r,userId:a}),e.executeAction(f.a,{authenticated:o,courseId:r,userId:a})}),Object(i.a)(function(e){var t=e.s12nStoreHasLoaded,r=e.courseStoreHasLoaded,n=e.sessionStoreHasLoaded,a=e.courseViewGradeStoreHasLoaded,o=e.progressStoreHasLoaded;return t&&r&&n&&a&&o}),Object(d.a)(function(e,t){var r=t.courseId,n=t.course,a=t.s12n,o=t.sessionId,i=s.a.get().id,u=s.a.isAuthenticatedUser(),c=a&&a.getId(),d=n.isVerificationEnabled(),l=e.getStore("CourseStore");e.executeAction(v.a,{userId:i}),l.isCumulativeGradePolicy()&&e.executeAction(E.a,{userId:i,courseId:r}),e.executeAction(m.a,{authenticated:u,userId:i,courseId:r,isCourseVerificationEnabled:d,s12nId:c}),e.executeAction(I.b,{courseId:r,userId:i,sessionId:o})}),Object(i.a)(function(e){var t;return e.verificationStoreHasLoaded}))(k),T=function LegacyDataFetch(e){var t=e.children,r=e.isLegacyDataLoaded;return o.a.createElement("div",{className:"rc-LegacyDataFetch"},o.a.createElement(M,null,t),!r&&o.a.createElement(F.a,{height:512},o.a.createElement(U.a,null)))};e.default=n()(Object(u.a)([P.a],function(e){return{isLegacyDataLoaded:e.hasLoaded()}}))(T)},NBfQ:function(module,e,t){"use strict";var r=t("S+eF"),n=t.n(r),a=t("sQ/U"),o=t("fghW"),s=t("xiyk"),i=t("ycgD");e.a=function(e,t){if(e.getStore(o.a).hasLoaded())return n()();var r;return Object(s.d)(t,a.a.get().id).then(function(e){var t=(r=e).elements,o=t&&t[0];return o&&a.a.isAuthenticatedUser()?Object(i.a)(o.id,!0):n()()}).then(function(t){return e.dispatch("LOAD_S12N",{rawS12ns:r,rawOwnership:t}),{rawS12ns:r,rawOwnership:t}})}},NO4R:function(module,e,t){"use strict";var r=t("S+eF"),n=t.n(r),a=t("BYIE"),o=t("qgMw"),s=t("lqQ6");e.a=function(e,t){if(e.getStore(o.a).hasLoaded())return n()();if(!t)return n.a.reject(new s.a("courseId must be provided."));return Object(a.a)(t).then(function(t){e.dispatch("LOAD_COURSE_SCHEDULE",t)})}},Re7p:function(module,e,t){"use strict";t.d(e,"a",function(){return u});var r=t("lSNA"),n=t.n(r),a=t("S+eF"),o=t.n(a),s=t("rvDt");function ownKeys(e,t){var r=Object.keys(e);if(Object.getOwnPropertySymbols){var n=Object.getOwnPropertySymbols(e);t&&(n=n.filter(function(t){return Object.getOwnPropertyDescriptor(e,t).enumerable})),r.push.apply(r,n)}return r}function _objectSpread(e){for(var t=1;t<arguments.length;t++){var r=null!=arguments[t]?arguments[t]:{};t%2?ownKeys(Object(r),!0).forEach(function(t){n()(e,t,r[t])}):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(r)):ownKeys(Object(r)).forEach(function(t){Object.defineProperty(e,t,Object.getOwnPropertyDescriptor(r,t))})}return e}var i={showHidden:!0,fields:["courseId","grade"],includes:{vcMembership:{fields:["certificateCode","grade","grantedAt"]},course:{fields:[]}}},u=function loadCertificateData(e,t){var r=t.courseId,n=t.userId,a;if(e.getStore("CertificateStore").hasLoaded())return o()();return(a=n?Object(s.a)(_objectSpread(_objectSpread({id:"".concat(n,"~").concat(r)},i),{},{rawData:!0})).then(function(t){e.dispatch("LOAD_MEMBERSHIPS",t)}):o()().then(function(){e.dispatch("LOAD_MEMBERSHIPS",null)})).done(),a}},TFmq:function(module,e,t){"use strict";var r=t("S+eF"),n=t.n(r),a=t("sQ/U"),o=t("tPFS"),s=t("D87r");e.a=function(e,t){if(e.getStore(o.a).hasLoaded())return n()();return a.a.isAuthenticatedUser()?Object(s.a)(t).then(function(t){t.elements&&t.elements.length&&e.dispatch("LOAD_HOME_PROGRESS",t.elements[0])}).fail(function(){e.dispatch("LOAD_HOME_PROGRESS",{modulesCompleted:[],modulesPassed:[]})}):(e.dispatch("LOAD_HOME_PROGRESS",{modulesCompleted:[],modulesPassed:[]}),n()())}},Vu1r:function(module,e,t){"use strict";var r=t("S+eF"),n=t.n(r),a=t("fw5G"),o=t.n(a),s=t("DnuM"),i=Object(s.a)("/api/onDemandCoursePresentGrades.v1",{type:"rest"}),u=function getPresentGrade(e){var t=e.userId,r=e.courseId,a=new o.a("/".concat(t,"~").concat(r)).addQueryParam("fields","grade,relevantItems,passingStateForecast");return n()(i.get(a.toString()))};e.a={getPresentGrade:u}},"b+2U":function(module,e,t){"use strict";t.d(e,"a",function(){return s}),t.d(e,"b",function(){return i});var r=t("S+eF"),n=t.n(r),a=t("jihO"),o=t("xPfO"),s=function getCurrentSession(e,t){var r=t.courseSlug;if(e.getStore(o.a).hasLoaded())return n()();return a.d(r).then(function(t){e.dispatch("LOAD_SESSION",t||null)}).fail(function(e){throw e})},i=function updateEnrollableAndFollowingSessions(e,t){var r=t.courseId,s=t.currentSessionId;if(e.getStore(o.a).hasLoaded())return n()();return a.e(r,s).then(function(t){(t.getUpcomingSession()||t.getFollowingSession())&&e.dispatch("LOAD_UPCOMING_AND_FOLLOWING_SESSIONS",{upcomingSession:t.getUpcomingSession(),followingSession:t.getFollowingSession()})}).fail(function(e){throw e})},u=function getAllSessions(e,t){if(e.getStore(o.a).hasLoaded())return n()();return a.c(t).then(function(t){e.dispatch("LOAD_ALL_SESSIONS",t)})}},dgIx:function(module,e,t){"use strict";t.d(e,"a",function(){return c}),t.d(e,"b",function(){return d});var r=t("S+eF"),n=t.n(r),a=t("F/us"),o=t.n(a),s=t("yFL5"),i=t("6p3O"),u=t("Aw3H"),c=function loadUserGroupsForCourse(e,t){var r=t.courseId,a=t.userId;if(e.getStore("GroupSettingStore").hasLoaded())return n()();return s.a.myCourseGroupsWithSettings(a,r).then(function(t){var r=o()(t.linked["groupSettings.v1"]).map(function(e){return new u.a(e)}),n=t.linked["groups.v1"].map(function(e){return new i.a(e)}),a=t.elements;e.dispatch("LOADED_COURSE_GROUPS",{groups:n,groupSettings:r,groupMemberships:a})}).fail(function(t){e.dispatch("LOADED_COURSE_GROUPS",{})})},d=function loadUserSessionGroupForCourse(e,t){var r=t.courseId,a=t.userId,o=t.sessionId;if(e.getStore("GroupSettingStore").hasSessionGroupLoaded())return n()();return s.a.getCourseSessionGroup(a,r,o).then(function(t){var r=t.elements[0];e.dispatch("LOADED_SESSION_GROUP",{sessionGroup:r})}).fail(function(t){e.dispatch("LOADED_SESSION_GROUP",{})})}},iTPM:function(module,e,t){"use strict";t.d(e,"a",function(){return l});var r=t("S+eF"),n=t.n(r),a=t("NBfQ"),o=t("sroZ"),s=t("NO4R"),i=t("hw75"),u=t("wbHF"),c=t("TFmq"),d=t("8c4I"),l=function loadComputedModels(e,t){var r=t.courseSlug,l=t.courseId;if(e.getStore(d.a).hasLoaded())return n()();return n.a.all([Object(o.a)(e),Object(a.a)(e,l),Object(i.a)(e,r),Object(s.a)(e,l),Object(u.a)(e,l),Object(c.a)(e,l)]).then(function(){e.dispatch("LOAD_COMPUTED_MODELS")})}},knci:function(module,e,t){"use strict";var r=t("VbXa"),n=t.n(r),a=t("q1tI"),o=t.n(a),s=t("juwT"),i=t("+LJP"),u=t("lngd"),c=function(e){function CourseUnauthorized(){return e.apply(this,arguments)||this}n()(CourseUnauthorized,e);var t=CourseUnauthorized.prototype;return t.componentDidMount=function componentDidMount(){var e=this.props.courseSlug;s.a.setLocation("/learn/".concat(e))},t.render=function render(){return o.a.createElement("div",{className:"align-horizontal-center"},o.a.createElement(u.a,null))},CourseUnauthorized}(o.a.Component);e.a=Object(i.a)(function(e,t){return{courseSlug:e.params.courseSlug}})(c)},pkWi:function(module,e,t){"use strict";var r=t("S+eF"),n=t.n(r),a=t("uYOU");e.a=function(e,t,r,o){if(r&&e){var s=n.a.all([Object(a.a)(e,t,!0)]).spread(function(e){var t;return{isProductVerificationEnabled:r,productOwnership:e,s12nId:o}},function(){return null});return s.done(),s}var i=n()(null);return i.done(),i}},rvDt:function(module,e,t){"use strict";var r=t("F/us"),n=t.n(r),a=t("DAK3"),o=t("oCg5"),s=t("7eiT"),i=t("FOnF");e.a=function(){var e=arguments.length>0&&void 0!==arguments[0]?arguments[0]:{};return Object(a.a)(e).then(function(t){if(t.linked&&t.linked["onDemandSessions.v1"]&&t.linked["onDemandSessionMemberships.v1"]){var r=n()(t.linked["onDemandSessions.v1"]).groupBy("courseId"),a=n()(t.linked["onDemandSessionMemberships.v1"]).groupBy("sessionId"),u=Object.keys(a);t.elements.forEach(function(e){var t=r[e.courseId]||[];if(t.length){var n=t.filter(function(e){return u.indexOf(e.id)>=0});if(n.length){var o=new i.b(n).getLastSession();e.onDemandSessionId=o.id,e.onDemandSessionMemberships=n.map(function(e){return a[e.id]})}}})}if(t.linked&&t.linked["v1Details.v1"]&&(t.linked["courses.v1"]=n()(t.linked["courses.v1"]).map(function(e){if("v1.session"===e.courseType||"v1.capstone"===e.courseType){e.v1Details=e.id;var r=n()(t.linked["v1Sessions.v1"]).reduce(function(t,r){return r.courseId===e.id&&t.push(r.id.toString()),t},[]);e.v1Sessions=r}return e})),t.linked&&t.linked["v2Details.v1"]&&(t.linked["courses.v1"]=n()(t.linked["courses.v1"]).map(function(e){return"v2.ondemand"===e.courseType&&(e.v2Details=n()(t.linked["v2Details.v1"]).findWhere({id:e.id})),e})),t.linked&&t.linked["vcMemberships.v1"]){var c=n()(t.linked["vcMemberships.v1"]).pluck("id");t.elements=n()(t.elements).map(function(e){return n()(c).contains(e.id)&&(e.vcMembershipId=e.id),e})}if(t.linked&&t.linked["courses.v1"]){var d=n()(t.linked["courses.v1"]).pluck("id");t.elements=n()(t.elements).chain().filter(function(e){return n()(d).contains(e.courseId)}).compact().value()}if(t.linked&&t.linked["signatureTrackProfiles.v1"]&&n()(t.elements).each(function(e){e.signatureTrackProfile=e.userId}),e.rawData)return t;if(e.withPaging)return{elements:Object(o.a)(s.a.prototype.resourceName,t),paging:t.paging};return Object(o.a)(s.a.prototype.resourceName,t)}).fail(function(t){if(e.rawData)return null;return new s.a})}},sArp:function(module,e,t){"use strict";t.d(e,"a",function(){return u});var r=t("S+eF"),n=t.n(r),a=t("fw5G"),o=t.n(a),s=t("TSOT"),i=Object(s.a)("/api/onDemandCourseSchedules.v1"),u=function getCourseSchedule(e){var t=new o.a(e).addQueryParam("fields","defaultSchedule");return n()(i.get(t.toString()))}},sroZ:function(module,e,t){"use strict";var r=t("S+eF"),n=t.n(r),a=t("DnuM"),o=t("IG+7");e.a=function(e){var t=Object(a.a)("/api/domains.v1",{type:"rest"});if(void 0!==e.getStore(o.a).domains)return n()();return n()(t.get("?fields=id,name")).then(function(t){e.dispatch("LOAD_DOMAINS",t.elements)})}},wDa7:function(module,e,t){"use strict";var r=t("S+eF"),n=t.n(r),a=t("TSOT"),o=t("fw5G"),s=t.n(o),i=t("sQ/U"),u=Object(a.a)("/api/onDemandDeadlineSettings.v1",{type:"rest"}),c={getStartTime:function getStartTime(e){var t=(new s.a).addQueryParam("q","byUserAndCourse").addQueryParam("userId",i.a.get().id).addQueryParam("courseId",e).toString();return n()(u.get(t)).fail(function(e){console.error(e)})},sendStartTime:function sendStartTime(e,t){var r={data:{userId:i.a.get().id,courseId:t,start:Date.now(),isEnabled:e}};return n()(u.post("",r))},disableDeadlines:function disableDeadlines(e){return c.sendStartTime(!1,e)},getResetPreview:function getResetPreview(e,t){var r=(new s.a).addQueryParam("q","extendPreview").addQueryParam("userId",i.a.get().id).addQueryParam("courseId",e).addQueryParam("extendedAt",Date.now()).toString();n()(u.get(r)).then(t).fail(function(e){console.error(e)}).done()},resetDeadlines:function resetDeadlines(e){var t={data:{userId:i.a.get().id,courseId:e,extendedAt:Date.now()}},r=(new s.a).addQueryParam("action","extend").toString();return n()(u.post(r,t))}};e.a=c;var d=c.getStartTime,l=c.sendStartTime,f=c.disableDeadlines,S=c.getResetPreview,p=c.resetDeadlines},wbHF:function(module,e,t){"use strict";var r=t("S+eF"),n=t.n(r),a=t("fw5G"),o=t.n(a),s=t("TSOT");e.a=function(e,t){var r=Object(s.a)("/api/onDemandReferences.v1",{type:"rest"}),a=(new o.a).addQueryParam("courseId",t).addQueryParam("q","courseListed").addQueryParam("fields","name,shortId,slug,content").addQueryParam("includes","assets");return n()(r.get(a.toString())).then(function(t){e.dispatch("LOAD_REFERENCES_LIST",t.elements)})}}}]);
//# sourceMappingURL=11.a4524dd7175709f67607.js.map