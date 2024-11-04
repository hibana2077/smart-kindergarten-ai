'use client';

import React, { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle, CardDescription, CardFooter } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Legend } from 'recharts';
import { FileText, Users, Book, Star, Calendar, Tag } from 'lucide-react';

// 模擬數據
const lessonPlans = [
  {
    id: 1,
    title: "春天色彩探索",
    type: "美術",
    date: "2024-03-15",
    tags: ["色彩認知", "創意表現"],
    status: "已完成"
  },
  {
    id: 2,
    title: "基礎幾何形狀",
    type: "認知",
    date: "2024-03-20",
    tags: ["幾何認知", "手部協調"],
    status: "進行中"
  },
  {
    id: 3,
    title: "團體創作活動",
    type: "綜合",
    date: "2024-03-25",
    tags: ["社交表現", "創意表現", "專注力"],
    status: "待開始"
  }
];

const performanceData = [
  { skill: '色彩認知', value: 85, fullmark: 100 },
  { skill: '幾何認知', value: 78, fullmark: 100 },
  { skill: '創意表現', value: 90, fullmark: 100 },
  { skill: '手部協調', value: 82, fullmark: 100 },
  { skill: '專注力', value: 75, fullmark: 100 },
  { skill: '社交表現', value: 88, fullmark: 100 },
];

const monthlyProgress = [
  { month: '1月', average: 82 },
  { month: '2月', average: 85 },
  { month: '3月', average: 88 },
  { month: '4月', average: 86 }
];

// interface LessonPlan {
//   id: number;
//   title: string;
//   type: string;
//   date: string;
//   tags: string[];
//   status: string;
// }

const LessonPlanCard = ({ plan }) => (
  <Card>
    <CardHeader>
      <div className="flex justify-between items-start">
        <div>
          <CardTitle className="text-lg">{plan.title}</CardTitle>
          <CardDescription>{plan.type}</CardDescription>
        </div>
        <Badge 
          variant={
            plan.status === "已完成" ? "default" : 
            plan.status === "進行中" ? "secondary" : 
            "outline"
          }
        >
          {plan.status}
        </Badge>
      </div>
    </CardHeader>
    <CardContent>
      <div className="flex items-center gap-2 mb-3">
        <Calendar className="w-4 h-4 text-gray-500" />
        <span className="text-sm text-gray-600">{plan.date}</span>
      </div>
      <div className="flex flex-wrap gap-2">
        {plan.tags.map((tag, index) => (
          <Badge key={index} variant="outline" className="bg-blue-50">
            <Tag className="w-3 h-3 mr-1" />
            {tag}
          </Badge>
        ))}
      </div>
    </CardContent>
  </Card>
);

const ClassDetailManagement = () => {
  const classInfo = {
    name: "向日葵班",
    year: "2024",
    description: "幼幼班",
    studentCount: 15
  };

  return (
    <div className="p-6 max-w-7xl mx-auto">
      {/* 班級基本信息 */}
      <div className="flex justify-between items-center mb-6">
        <div>
          <h1 className="text-2xl font-bold mb-2">{classInfo.name}</h1>
          <div className="flex items-center gap-4">
            <Badge variant="outline">{classInfo.year}年</Badge>
            <span className="text-gray-500">{classInfo.description}</span>
            <div className="flex items-center gap-1">
              <Users className="w-4 h-4" />
              <span>{classInfo.studentCount} 位學生</span>
            </div>
          </div>
        </div>
      </div>

      {/* 主要內容區 */}
      <Tabs defaultValue="lessons" className="space-y-6">
        <TabsList>
          <TabsTrigger value="lessons" className="flex items-center gap-2">
            <Book className="w-4 h-4" />
            教案管理
          </TabsTrigger>
          <TabsTrigger value="performance" className="flex items-center gap-2">
            <Star className="w-4 h-4" />
            學習成效
          </TabsTrigger>
        </TabsList>

        {/* 教案管理內容 */}
        <TabsContent value="lessons">
          <div className="space-y-6">
            <div className="flex justify-between items-center">
              <h2 className="text-xl font-semibold">教案列表</h2>
              <Button>
                <FileText className="w-4 h-4 mr-2" />
                生成新教案
              </Button>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              {lessonPlans.map(plan => (
                <LessonPlanCard key={plan.id} plan={plan} />
              ))}
            </div>
          </div>
        </TabsContent>

        {/* 學習成效內容 */}
        <TabsContent value="performance">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* 能力雷達圖 */}
            <Card>
              <CardHeader>
                <CardTitle>班級能力分布</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="h-80">
                  <ResponsiveContainer width="100%" height="100%">
                    <RadarChart data={performanceData}>
                      <PolarGrid />
                      <PolarAngleAxis dataKey="skill" />
                      <PolarRadiusAxis angle={30} domain={[0, 100]} />
                      <Radar
                        name="能力值"
                        dataKey="value"
                        stroke="#2563eb"
                        fill="#2563eb"
                        fillOpacity={0.6}
                      />
                    </RadarChart>
                  </ResponsiveContainer>
                </div>
              </CardContent>
            </Card>

            {/* 月度趨勢圖 */}
            <Card>
              <CardHeader>
                <CardTitle>學習進度趨勢</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="h-80">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={monthlyProgress}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="month" />
                      <YAxis domain={[0, 100]} />
                      <Tooltip />
                      <Legend />
                      <Line
                        type="monotone"
                        dataKey="average"
                        stroke="#2563eb"
                        strokeWidth={2}
                      />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </CardContent>
            </Card>

            {/* 詳細數據表格 */}
            <Card className="md:col-span-2">
              <CardHeader>
                <CardTitle>各項能力詳細數據</CardTitle>
              </CardHeader>
              <CardContent>
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead>能力項目</TableHead>
                      <TableHead>平均分數</TableHead>
                      <TableHead>最高分數</TableHead>
                      <TableHead>最低分數</TableHead>
                      <TableHead>進步幅度</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {performanceData.map((item) => (
                      <TableRow key={item.skill}>
                        <TableCell className="font-medium">{item.skill}</TableCell>
                        <TableCell>{item.value}</TableCell>
                        <TableCell>{item.value + 5}</TableCell>
                        <TableCell>{item.value - 8}</TableCell>
                        <TableCell className="text-green-600">+3.5%</TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </CardContent>
            </Card>
          </div>
        </TabsContent>
      </Tabs>
    </div>
  );
};

export default ClassDetailManagement;